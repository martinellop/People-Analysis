import argparse
import time
import os
import shutil
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


debug_mode=False
def debugPrint(*infos):
    if not debug_mode:
        return
    if len(infos) < 1:
        return
    res = "[DEBUG] "
    for i in infos:
        res = res + " " + str(i)
    print(str(res))


def detect(settings):
    source, weights, imgsz, trace = settings.source, settings.weights, settings.img_size, not settings.no_trace
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    # Initialize
    set_logging()
    device = select_device(settings.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, settings.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    #debugPrint("classes:",names)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    frame_number = -1
    old_path = ""
    last_frame_saved = -1

    frames_bb = {}

    for path, img, im0s, vid_cap in dataset:

        if frame_number >= 0 and old_path != path:
            #video changed (multiple videos)
            frame_number = -1
            last_frame_saved = -1
        old_path = path
        frame_number += 1

        if dataset.mode == 'video':
            if last_frame_saved >= 0 and frame_number - last_frame_saved < settings.video_crop_min_frame_interval:
                #in this case, we want to skip some frames (in order to gain diversity in output).
                #let's skip them before sending to the model, improving efficiency.
                continue

        debugPrint(f"\nframe number: {frame_number}")
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=settings.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=settings.augment)[0]
        t2 = time_synchronized()
        #debugPrint("boxes before NMS:",len(pred[0]))

        # Apply NMS
        pred = non_max_suppression(pred, settings.conf_thres, settings.iou_thres, classes=settings.classes, agnostic=settings.agnostic_nms)
        t3 = time_synchronized()
        #debugPrint("boxes after NMS:",len(pred[0]))
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                #printing
                for c in det[:,-1].unique():
                    n = (det[:, -1] == c).sum() #detections found 
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                correct_labels=0
                for *xyxy, conf, cls in reversed(det):

                    # Aggiungo lo 0 della classe
                    bb = torch.tensor(xyxy).view(1, 4)
                    zero = torch.zeros((1,1))
                    bb = torch.cat((zero, bb), 1)

                    if frame_number in frames_bb:
                        frames_bb[frame_number] = torch.cat((frames_bb[frame_number], bb))
                    else:
                        frames_bb[frame_number] = bb


                last_frame_saved = frame_number

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    print(f'Done. ({time.time() - t0:.3f}s)')

    return frames_bb


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #print(f"c1:{c1}, c2:{c2}")
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


def extract_bb(source, framerate, weights, minarea=15000):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--video_crop_min_area', type=int, default=0, help='minimum number of pixels in detected box for a crop')
    parser.add_argument('--video_crop_max_area', type=int, default=-1, help='maximum number of pixels in detected box for a crop')
    parser.add_argument('--video_crop_min_frame_interval', type=int, default=1, help='minimum number of frames to be skipped from last crop-saved frame')

    settings = parser.parse_args(["--source", source,
                                  "--weights", weights,
                                  "--class", "0",
                                  "--video_crop_min_area", str(minarea),
                                  "--video_crop_min_frame_interval", str(framerate)])

    return detect(settings)

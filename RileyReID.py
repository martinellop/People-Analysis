import os
import cv2
import numpy as np
import random
import time
import torch
import torchvision.transforms as T
from torchvision.ops import roi_pool

from PeopleDetector.utils.datasets import LoadImages
from PeopleDetector.PeopleDetector import extract_bb
from ReID.common.PeopleDB import PeopleDB
from ReID.common.distances import L2_distance, Cosine_distance
from ReID.common.MOTutils import plot_one_box
from ReID.deep.model import ReIDModel


# Faccio questa aggiunta per evitare un problema nel import del modello di yolo
import sys
module_folder = os.path.abspath(os.path.join("PeopleDetector"))
sys.path.insert(0, module_folder)


class HyperParams:
    def __init__(self, target_resolution=(128,64), frame_stride=15, dist_function=L2_distance, threshold:float=1.0, max_descr_per_id:int=3, db_memory:int=20*10):
        self.target_res = target_resolution
        self.frame_stride = frame_stride
        self.dist_function = dist_function
        self.threshold = threshold
        self.max_descr_per_id = max_descr_per_id
        self.frame_memory = db_memory


def analyze_video(video_path, output_path, model, yolomodel, preprocess, device, hyperPrms):
    #video_resolution = (852,480)
    #video_resolution = (960,540)
    video_resolution = (1920,1080)
    video = LoadImages(video_path,video_resolution[0])
    out_framerate = 10

    position_history = {}

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), out_framerate, video_resolution)

    db = PeopleDB(hyperPrms.dist_function, hyperPrms.threshold, int(hyperPrms.frame_memory / hyperPrms.frame_stride),
                  hyperPrms.max_descr_per_id, device=device)
    ncolors = 255
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(ncolors)]
    current_frame = -1

    # Uso Yolo per estrarre tutte le Bounding box
    frames_bb = extract_bb(video_path, hyperprms.frame_stride, weights=yolomodel)
    analyzed_frames = frames_bb.keys()

    # Analizzo le varie BB
    for path, img, im0s, vid_cap in video:

        current_frame += 1
        if current_frame not in analyzed_frames:
            continue

        boxes = frames_bb[current_frame]
        frame_tensor = torch.from_numpy(im0s).to(dtype=torch.float, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        crops = roi_pool(frame_tensor, boxes, hyperPrms.target_res).to(device)
        crops = preprocess(crops)

        target_ids = torch.zeros(size=(0, 1), device=device, dtype=torch.long)
        descriptors = model(crops)
        print("descriptors:", descriptors.shape)
        for i in range(descriptors.shape[0]):
            descr = descriptors[i]
            target_id, new_one = db.Get_ID(descr)
            id_ = torch.Tensor((target_id,)).to(dtype=torch.long, device=device).reshape(1, 1)
            target_ids = torch.cat((target_ids, id_))
        db.Update_Frame()

        frame = im0s
        for i in range(boxes.shape[0]):
            id = int(target_ids[i])

            # Calcolo posizione media
            mean = (int((boxes[i,1]+boxes[i,3])/2), int((boxes[i,2]+boxes[i,4])/2))
            try:
                position_history[id].append(mean)
            except KeyError:
                position_history[id] = [mean]

            plot_one_box(boxes[i, 1:5], frame, label=str(id), color=colors[id % ncolors])
            plot_history(position_history[id], frame, color=colors[id % ncolors])

        video_writer.write(frame)

    video_writer.release()


def plot_history(history, frame, color):
    for i in range(len(history)-1):
        cv2.line(frame, history[i], history[i+1], color, 3)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperprms = HyperParams(threshold=0.4, target_resolution=(224, 224), dist_function=Cosine_distance)
    model = ReIDModel(model="resnet18").to(device)
    weights_path = "ReID/deep/results/training6/model.bin"
    yolo_weights = "/Users/infopz/Not_iCloud/People-Analyzer/PeopleDetector/yolov7-tiny.pt"
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)

    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    videopath = "inputs/videos/example_MOT20.mp4"
    outputpath = "out.avi"

    model.eval()
    with torch.no_grad():
        analyze_video(videopath, outputpath, model, yolo_weights, transform, device, hyperprms)
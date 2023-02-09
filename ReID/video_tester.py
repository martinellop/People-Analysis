import torch
import torch.nn as nn
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='True'
import sys
import random
import cv2
from torchvision.ops import roi_pool
import torchvision.transforms as T

from common.MOTutils import MOTannotation,plot_one_box
from common.PeopleDB import PeopleDB
from common.distances import L2_distance, Cosine_distance

from HOG.HogDescriber import HOGModel

from deep.model import ReIDModel

module_folder = os.path.abspath(os.path.join("PeopleDetector", "utils"))
print(module_folder)
sys.path.insert(0, module_folder)
from datasets import LoadImages


class HyperParams:
    def __init__(self, target_resolution=(128,64), frame_stride=1, dist_function=L2_distance, threshold:float=1.0, max_descr_per_id:int=3, db_memory:int=20*10):
        self.target_res = target_resolution
        self.frame_stride = frame_stride
        self.dist_function = dist_function
        self.threshold = threshold
        self.max_descr_per_id = max_descr_per_id
        self.frame_memory = db_memory

def Evaluate_On_MOTSynth(model:nn.Module, hyperPrms:HyperParams, preprocess:T, device:torch.device, visualize:bool=True, save_output:bool=False, max_time:float=-1):
    
    # example clip data:
    data_folder = os.path.join(".", "inputs", "videos", "512")
    video_path = os.path.join(data_folder, "512.mp4")
    annotations_path = os.path.join(data_folder, "gt", "gt.txt")
    video = LoadImages(video_path,1920)
    framerate = 20

    if save_output:
        output_path = os.path.join(".", "outputs", "videos", "512.mp4")
        vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), framerate, (1920, 1080))


    max_frame = round(max_time * framerate) if max_time > 0 else -1

    db = PeopleDB(hyperPrms.dist_function, hyperPrms.threshold, int(hyperPrms.frame_memory / hyperPrms.frame_stride), hyperPrms.max_descr_per_id, device=device)
    data = MOTannotation(annotations_path)
    ncolors = 255
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(ncolors)]
    current_frame = -1

    for path, img, im0s, vid_cap in video:
        
        t0 = time.time()
        current_frame+=1

        if current_frame % hyperPrms.frame_stride != 0:
            continue
        boxes = torch.zeros(size=(0,5),device=device, dtype=torch.float)
        mot_ids = torch.zeros(size=(0,1),device=device, dtype=torch.long)
        target_ids = torch.zeros(size=(0,1),device=device, dtype=torch.long)
        for el in data[current_frame+1]: #in annotations frame number are in base 1
            b = torch.Tensor((0,el.bb_left_,el.bb_top_, el.bb_left_ + el.bb_width_, el.bb_top_ + el.bb_height_)).to(dtype=torch.float, device=device).reshape(1,5)
            boxes = torch.cat((boxes,b))
            id = torch.Tensor((el.id_,)).to(dtype=torch.long, device=device).reshape(1,1)
            mot_ids = torch.cat((mot_ids, id))

        t1 = time.time()
        #print(boxes.shape, boxes)
        frame_tensor = torch.from_numpy(im0s).to(dtype=torch.float, device=device).permute(2,0,1).unsqueeze(0) / 255.0
        crops = roi_pool(frame_tensor,boxes, hyperPrms.target_res).to(device)
        crops = preprocess(crops)

        t2 = time.time()
        #print("crops after roiPooling:", crops.shape)
        descriptors = model(crops)
        print("descriptors:", descriptors.shape)
        for i in range(descriptors.shape[0]):
            descr = descriptors[i]
            target_id, new_one = db.Get_ID(descr)
            id_ = torch.Tensor((target_id,)).to(dtype=torch.long, device=device).reshape(1,1)
            target_ids = torch.cat((target_ids, id_))
            #report = f"+ Created {target_id} at frame {current_frame}" if new_one else f"Recognized {target_id} at frame {current_frame}"
            #print(report)
        t3 = time.time()
        db.Update_Frame()
        t4 = time.time()
        s = f'frame {current_frame} BB mgmt: {round((t1 - t0)*1000)}ms; ROIpooling: {round((t2 - t1)*1000)}ms; Descr gen: {round((t3 - t2)*1000)}ms; frameShift: {round((t4 - t3)*1000)}ms'

        frame = im0s
        for i in range(boxes.shape[0]):
            id = int(target_ids[i])
            plot_one_box(boxes[i,1:5],frame,label=str(id),color=colors[id%ncolors])

        if save_output:
            vid_writer.write(frame)

        if visualize:
            cv2.imshow("img", frame)
            cv2.waitKey(1)  # 1 millisecond
        t5 =time.time()


        if visualize:
            s = s + f"; plotting: {round((t5 - t4)*1000)}ms"

        s = s + f". TOTAL:{round((t5 - t0)*1000)}ms."
        print(s)    

        if(max_frame > 0 and current_frame >= max_frame-1):
            break
    db.Clear()



if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_hog_descr = False   #setting to True means to use the hog model, if False it will be used the deep model.
    
    if use_hog_descr:
        hyperprms = HyperParams(threshold=2.5,target_resolution=(128,64))
        model = HOGModel().to(device)
    else:
        #let's load the deep model
        hyperprms = HyperParams(threshold=0.4,target_resolution=(224,224), dist_function=Cosine_distance)
        model = ReIDModel(model="resnet18").to(device)
        weights_path = "D:\\Data\\University\\People-Analysis\\ReID\\deep\\results\\training6\\model.bin"
        weights = torch.load(weights_path)
        model.load_state_dict(weights)
    
    transform = T.Compose([
        #T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    with torch.no_grad():
        Evaluate_On_MOTSynth(model, hyperprms, transform, device, visualize=False, save_output=True, max_time=-1)
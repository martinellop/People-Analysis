import os
import torch
import json
import sys
#from PeopleDetector.utils.datasets import LoadImages
import numpy as np
import cv2
import time
import math
import threading
import logging

module_folder = os.path.abspath(os.path.join("PeopleDetector", "utils"))
#print(module_folder)
sys.path.insert(0, module_folder)
from datasets import LoadImages

root_dir = "D:\\Data\\University\\MOTSynth"

videos_dir = os.path.join(root_dir,"MOTSynth_videos")
annotations_dir = os.path.join(root_dir,"annotations_COCO_style")
crops_dir = os.path.join(root_dir,"crops_2")      #output dir
framerate = 20
video_res=(1920,1080)


# SETTINGS
extraction_period = 5   # a value 10 means "extract crops form one frame every 10 frames (aka every 0.5 seconds)" 
avoid_blurred_images = False
avoid_night_clips = True
minim_crop_area = 6000      # crops w*h must be at least this value in order to get saved
minim_dim = 35              # minimal lenght in pixel of each dimension.
minim_person_area = 1500    # minimum amount of pixels in which it is possible to see the target (e.g. behind a wall area is 0)
isolate_clips = False

n_workers = 1               # how many threads do you want to work in parallel?

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def process_crop(annotation, frame:np.ndarray, image_id:int, clip_name:str): 
    if avoid_blurred_images and annotation['is_blurred'] > 0:
        return False

    if annotation['area'] < minim_person_area:
        return False

    bbox = annotation['bbox']
    if bbox[2] * bbox[3] < minim_crop_area:
        return False
    if bbox[2] < minim_dim or bbox[3] < minim_dim:
        return False

    model_id = annotation['model_id'] #it's a string 

    if not isolate_clips:
        attributes = annotation['attributes']
        name = f"{model_id:03d}_"
        for el in attributes:
            if el > 99:
                return False # we store each attribute in 2 digits
            name += f'{el:02d}'
        #print(f"mod_id: {model_id}, attributes: {attributes} --> name: {name}")
        path = os.path.join(crops_dir, name)
        filename = os.path.join(path, f"{image_id:07d}.jpg")
    else:
        path = os.path.join(crops_dir, f"{model_id:03d}_{clip_name}")
        filename = os.path.join(path, f"{image_id:07d}.jpg")

    if not os.path.isdir(path):
        os.mkdir(path)
    crop = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    
    if not os.path.exists(filename):
        cv2.imwrite(filename,crop)
    return True


def process_videos(thread_id:int, videos):
    processed = 0
    for v in videos:
        video = LoadImages(os.path.join(videos_dir,v) ,video_res[0], verbose=False)
        short_name = v.replace(".mp4", "")

        ann_file = short_name +  ".json"
        annotations = json.load(open(os.path.join(annotations_dir,ann_file)))

        if avoid_night_clips:
            if annotations['info']['is_night'] == 1:
                processed += 1
                logging.info(f"Thread_{thread_id}: SKIPPED video {short_name} since during night. Total: {round(processed * 100 / (len(videos)))}% done.")
                continue

        annotations = annotations['annotations']    # we're interested just to where there are object annotations.

        image_id = int(short_name) * 10000 - 1

        ann_idx = 0
        #print(annotations[0])
        logging.info(f"Thread_{thread_id}: STARTING video: {short_name}.")
        t1 = time.time()    
        saved_crops = 0

        current_frame = -1
        finished = False
        for path, img, im0s, vid_cap in video:
            current_frame +=1
            image_id += 1
            if(current_frame % extraction_period != 0):
                continue
            
            while annotations[ann_idx]['image_id'] < image_id:
                ann_idx+=1
                if ann_idx == len(annotations):
                    finished = True
                    break

            if finished:
                break    

            while annotations[ann_idx]['image_id'] == image_id:
                if process_crop(annotations[ann_idx], im0s, image_id, short_name):
                    saved_crops +=1
                ann_idx+=1
                if ann_idx == len(annotations):
                    finished = True
                    break

            if finished:
                break
        processed += 1        
        logging.info(f"Thread_{thread_id}: DONE video: {short_name}. Saved {saved_crops} crops in {round(time.time() - t1)} seconds. Total: {round(processed * 100 / (len(videos)))}% done.")
        



if __name__=='__main__':
    videos = get_files_from_folder(videos_dir)
    annotations = get_files_from_folder(annotations_dir)
    logging.basicConfig(level=logging.INFO)
    logging.info(f"clips to be processed: {len(videos)}")

    if not os.path.isdir(crops_dir):
        os.mkdir(crops_dir)

    t0 = time.time()

    n_workers = min(n_workers, len(videos))
    threads = []

    task_list = []
    remaining = len(videos)
    starting_idx = 0
    for w in range(n_workers):
        count = math.ceil(remaining / (n_workers-w))
        remaining -= count
        v_list = videos[starting_idx: starting_idx+count]
        task_list.append(v_list)
        starting_idx +=count

    for index in range(n_workers):
        x = threading.Thread(target=process_videos, args=(index, task_list[index]),daemon=True)
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    #print("TaskList:", task_list)
    #process_videos(videos)
    logging.info(f"--FINISHED in {round(time.time() - t0)} seconds; processed {len(videos)} videos.")


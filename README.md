# People-Analyzer



--> 🛑project in progress🛑 <--

## AIM: creation of a surveillance tool for public environments
Starting from video streams coming from security cameras, carry out detection, reID and therefore tracking of visible people, together with pose estimation and possibly instance segmentation. The operator will have the possibility to select a subject followed by the tracking system, obtain an visually improvemed image respect the original one and finally carry out a retrieval of the person in question from a given dataset, obtaining images better resembling the target.

## Main components:
### 1. Main Network
##### Services: People Detection, Tracking, Pose estimation [, Instance segmentation]
Neural network based on yolov7, whose last layers are removed and replaced with multiple heads, where each head is dedicated to a service.
The training would take place by disconnecting the gradient at the interface between yolo and heads.

### 2. Image enhancement pipeline
##### Services: Improving the quality of a low resolution image containing a person
Starting from a low resolution image, application of filters to remove noise and improve visual quality. Later, if possible, a generative model will be used for an image reconstruction with higher quality and resolution.

### 3. Retrieval Network
##### Services: recovery of images similar to the one provided
Neural network which, starting from an image containing a person/face, retrieves the most similar images within a previously provided dataset.



## Why this project? 
This project was developed as group project for the course of "Computer Vision & Cognitive Systems" done at the [University of Modena and Reggio Emilia](https://www.unimore.it/).

Group members:
- Pietro Martinello
- Leonardo Zini
- Giovanni Casari

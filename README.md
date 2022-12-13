# People-Analyzer



--> 🛑project in progress🛑 <--

## AIM: creation of a surveillance tool for public environments
Starting from video streams coming from security cameras, carry out detection, reID and therefore tracking of visible people. The operator will have the possibility to select a subject followed by the tracking system, obtain a visually improved image respect the original one and finally carry out a retrieval of the person in question from a given dataset, obtaining images better resembling the target.

## Data sources:
- MOTSynth for people tracking,
- LTFT for face tracking,
- CelebA for resolution enhancer network
- Eventually, video and pictures recorded by us

## Components:

### 0. Detector:
##### Services: People Detection
Pre-trained Yolov7 network used in inference to detect people in video frames.
The result of this stage would be cropped images of humans, that will be used in the next stages

### 1. Tracker
##### Services: ReID and tracking
The aim of this component is to perform re-identification of the same persons in different frames of the same video stream. It would be done by two alternative subcomponents, in order to compare results obtained by different techniques:
  - Classical method:
  ReID is done using as descriptor histograms of gradient over the person area.
  Note: to avoid interference of area not related to humans, instance segmentation is probably needed to select only specific pixels for gradient evaluation.
  - Deep method:
  ReID is done using a custom neural network, of which both design and training would be performed by us.
  The results of the two alternatives would be put in comparison and analysed.

### 2. Image enhancement pipeline
##### Services: Improving the quality of a low resolution image containing a person
Starting from a low resolution image, application of filters to remove noise and improve visual quality. Later, if possible, a generative model will be used for an image reconstruction with higher quality and resolution.

### 3. Retrieval Network
##### Services: recovery of images similar to the one provided
Classical or deep method which, starting from an image containing a person/face, retrieves the
most similar images within a previously provided dataset.



## Why this project? 
This project was developed as group project for the course of "Computer Vision & Cognitive Systems" done at the [University of Modena and Reggio Emilia](https://www.unimore.it/).

Group members:
- Pietro Martinello
- Leonardo Zini
- Giovanni Casari

# Tracking-and-People-Analysis

The scope of our project is to perform analysis about people's behaviour through a video
stream, like from a surveillance camera, gathering information about human poses,
distances, directions and making possible super-resolution on faces and licence plates, in
those circumstances in which the available resolution is not acceptable.


### 1. Topic: 
Tracking and People analysis,

distance estimation,

pose estimation,

super-resolution on faces and licence plates.

### 2. Data:
a) MotsynthCollect for people tracking,

b) LTFT for long-term face tracking,

c) CelebA and some video surveillance recorded by us.

### 3. Pipeline:
a) Tracking people and segment the shape of every person in the video

b) With the use of some classical image processing methods, estimate the pose
of the people and the distance between them.

c) Make it possible to allow the user to, when the video is in pause, to select an
area of the image, and in accordance with that region find, eventually, a face
or a license plate and, through the use of a generative autoencoder,
super-resoluted it.

### 4. Expected Result and measurements:
Evaluate the quality of the tracking by hand, the
distance comparing the result of our pipeline with the real distance and the
super-resolution comparing the generated one with the original at the same
resolution. We will tend to compare the result of our pipeline in each step with the
comparison of the result produced by our system with the same situation in the real
world.

# Football POV Image Stabilization and Reconstruction
Eddie Sasagawa, Jason Moericke, Dongha Kang
[[Paper](https://drive.google.com/open?id=1E_ioyBxfJyYDA63hRRQAdZKHdfP7ph5t)] [[Presentation](https://drive.google.com/open?id=15lDljNIgDdQeYQcK9gfZqfQe61rjwWjM)]


## Work
In this work, the objective is to embed a camera inside a football, and develop an image processing pipeline to de-spin the footage from the spinning camera, provide stable images, and reconstruct them.
The baseline we have chosen is work by Horita and his co-authors, called Experiencing the Ball’s POV for Ballistic Sports. The baseline work includes down-finding with mean pixel intensity, Image stitching method called graphcuts, and video synthesis with homography interpolation. The baseline work mounted the camera to the side of the football. However we propose to mount the camera slightly off of the front axis. This is to balance an enlarged field of view with a constant view of the football target.

## Result
Constructed video with frames interpolated.
![Alt Text](https://drive.google.com/uc?export=view&id=1wv-aMCNSLMLrYUy-fl3V-MdZODPgHUZ)



<!-- ## Installation
In this implemenation specifically, opencv version 3.4.2.16 is used
```
pip3 install opencv-python==3.4.2.16
pip3 install opencv-contrib-python==3.4.2.16
```
To run
```
cd $PATH
python3 Stereo_Reconstruction.py
``` -->

# Camera Calibration

1. Takes a video ("./20200208_football_throw01.mp4") and puts every 3rd frame image to a jpg file.
2. I tried SIFT, SURF, ORB and seems like ORB matches the best.
3. By printing value "h", will see homogenous matrix.

## From Here
1. Picture warping using warpPerspective function.
2. Trying to map with second order interpolation instead of linear interpolation.


## Update
1. distortion coefficient does not work properly at this point. (without any distortion coefficient, it does not have an error)



https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

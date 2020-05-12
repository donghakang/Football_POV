import cv2
import numpy as np
import os
import glob
import sys

from despin_video.rectification.camera_calibration import FisheyeCalibration

if __name__ == '__main__':
    images = glob.glob('/home/eddie/Documents/Courses/CSCI5561/Calibration/gopro*.png')

    fisheye_cal = FisheyeCalibration(load_coeffs=False)

    K, D, rvecs, tvecs = fisheye_cal.calibrate(images, save_matrices=True)

    vidcap = cv2.VideoCapture('20200208_football_throw02.mp4')
    # vidcap = cv2.VideoCapture('20200208_football_throw02.MP4')
    success,image = vidcap.read()
    count = 0
    while success:
        success,image = vidcap.read()
        if not success:
            break

        output = fisheye_cal.rectify(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        vertical = np.hstack((output, image))
        cv2.imshow("Undistorted  ||  Distorted", vertical)
        cv2.waitKey(50)
        if (count % 30 == 0):
            save_title = "./output/undistortion_" + str(count/30) + ".png"
            cv2.imwrite(save_title, vertical)
        count += 1

    cv2.destroyAllWindows()

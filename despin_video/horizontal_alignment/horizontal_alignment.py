import os
import glob
import shutil
import datetime
import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt

from keras.models import load_model

from despin_video.horizontal_alignment.correct_rotation import process_images
from despin_video.horizontal_alignment.utils import RotNetDataGenerator, crop_largest_rectangle, angle_error, rotate

class HorizonAlignment:
    def __init__(self, output_dir):
        thisdir = os.path.dirname(os.path.realpath(__file__))
        self._cnn_model = os.path.join(thisdir, 'rotnet_composite_resnet50_final.hdf5')

        self.my_results_dir = os.path.join(output_dir, 'horizon_alignment')

    def align(self, frame_bgr):
        # THIS IS GARBAGE.  LEAVING IT HERE TEMPORARILY
        # IN CASE WE REVISIT THIS APPROACH
        # convert to grayscale
        frame = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)

        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(frame,127,255,0)

        # calculate moments of binary image
        M = cv2.moments(thresh)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        #down_sample = cv2.resize(cv2.split(frame_bgr)[0], (self.frame_width // 4, self.frame_height // 4), 0, 0, cv2.INTER_NEAREST)
        blured = cv2.Sobel(frame,cv2.CV_8U,0,1,ksize=3)
        filter_x, filter_y = self.get_differential_filter()
        #my_blured = self.filter_image(frame, filter_y)
        #blured = cv2(frame,cv2.CV_8U,0,1,ksize=3)
        #blured_bw = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(img,(5,5),0)
        #edges = cv2.Canny(frame,50,150,apertureSize = 3)
        #edges = cv2.Canny(frame,200,600,apertureSize = 3)
        #edges = cv2.Canny(frame,200,600,apertureSize = 3)
        edges = cv2.Canny(blured,2000,6000,apertureSize = 5)
        # put text and highlight the center
        # cv2.circle(frame_bgr, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(frame_bgr, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #cv2.imshow('Frame ',edges)
        #cv2.waitKey(0)

        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        vx, vy, x0, y0  = cv2.fitLine(np.argwhere(edges == 255), 2, 0, 0.001, 0.001)  # 2 = CV_DIST_L2

        if lines is not None:
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(frame_bgr,(x1,y1),(x2,y2),(0,255,0),2)

        if vx == vx:
            m=2000
            # plt.line(frame_bgr, (y0-m*vy[0], x0-m*vx[0]), (y0+m*vy[0], x0+m*vx[0]) ,(0,255,0),2)
            # plt.circle(frame_bgr, (y0, x0), 5, (255, 100, 100), -1)
            cv2.line(frame_bgr, (y0-m*vy[0], x0-m*vx[0]), (y0+m*vy[0], x0+m*vx[0]) ,(0,255,0),2)
            cv2.circle(frame_bgr, (y0, x0), 5, (255, 100, 100), -1)

        # cv2.imshow('Frame ',frame_bgr)
        # cv2.waitKey()

        # cv2.imshow('Frame ',blured)
        # cv2.waitKey(0)
        # # #cv2.imshow('Frame',blured)
        # cv2.imshow('Frame ',edges)
        # cv2.waitKey(0)

        # plt.imshow(my_blured)
        # plt.show()
        plt.imshow(frame_bgr)
        plt.show()
        plt.imshow(blured)
        plt.show()
        plt.imshow(edges)
        plt.show()

        return aligned_image

    def align_cnn(self, input_dir, crop=False):
        # Clean out the directory
        if os.path.isdir(self.my_results_dir):
            shutil.rmtree(self.my_results_dir)
        os.mkdir(self.my_results_dir)

        #########
        input_path = input_dir #'./temp_prerotate_composites/' + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '/'
        output_path = self.my_results_dir #'./temp_postrotate_composites/' + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '/'

        # if not os.path.exists(input_path):
        #     os.makedirs(input_path)
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        # files = glob.glob(input_path)
        # for f in files:
        #     os.remove(f)

        # files = glob.glob(output_path)
        # for f in files:
        #     os.remove(f)

        # for comp, compname in zip(composites, compositenames):
        #     #print(os.path.join(input_path, compname, '.png'))
        #     cv2.imwrite(os.path.join(input_path, compname + '.png'), comp)

        print('Loading model...')
        model_location = load_model(self._cnn_model, custom_objects={'angle_error': angle_error})

        print('Rotating input image(s)...')
        batch_size = 4
        process_images(model_location, input_path, output_path,
                    batch_size, crop)

        # imgfilelist = [os.path.join(output_path, f)
        #                for f in os.listdir(output_path)]
        # complist = []
        # compnames = []
        # for imfile in imgfilelist:
        #     print('retrieving ' + imfile)
        #     complist.append(cv2.imread(imfile))
        #     compnames.append(os.path.splitext(os.path.basename(imfile))[0])

        # return complist, compnames
        return self.my_results_dir

#HorizonAlignment().align(cv2.imread('./images/IMG_2228 copy.png'))
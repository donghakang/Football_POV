import cv2
import numpy as np
import os
import sys

class FisheyeCalibration:
    CAMERA_MATRIX_FILENAME = 'camera_matrix'
    DIST_COEFF_FILENAME = 'distortion_coeffs'

    def __init__(self, load_coeffs=True):
        self._mydir = os.path.dirname(os.path.realpath(__file__))
        if load_coeffs:
            self._camera_matrix = np.load('{}/{}.npy'.format(self._mydir, self.CAMERA_MATRIX_FILENAME))
            self._distortion_coeffs = np.load('{}/{}.npy'.format(self._mydir, self.DIST_COEFF_FILENAME))
        else:
            self._camera_matrix = None
            self._distortion_coeffs = None

        self._checkerboard_inner_size = (6,8)
        self._rectify_map1 = None
        self._rectify_map2 = None

    def calibrate(self, imgfiles, save_matrices=False):
        '''
        input  : files to images of checker board (6x8 inner corners)
        output : K matrix - floating camera matrix
                 D matrix - distortion coefficient
        '''
        checkerboard_h, checkerboard_w = self._checkerboard_inner_size

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((1, checkerboard_h*checkerboard_w,3), np.float32)
        objp[0,:,:2] = np.mgrid[0:checkerboard_h,0:checkerboard_w].T.reshape(-1,2)
        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for fname in imgfiles:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self._checkerboard_inner_size, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                pattern_corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1), criteria)
                imgpoints.append(corners)

                # cv2.drawChessboardCorners(img, self._checkerboard_inner_size, pattern_corners, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        flags        = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        rvecs        = np.zeros((len(objpoints), 1, 1, 3))
        tvecs        = np.zeros((len(objpoints), 1, 1, 3))

        rms, K, D, rvecs, tvecs = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

        print('Camera Matrix:')
        print(K)
        print(' ')
        print('Distortion Coefficients:')
        print(D)

        if save_matrices:
            np.save('{}/{}'.format(self._mydir, self.CAMERA_MATRIX_FILENAME), K)
            np.save('{}/{}'.format(self._mydir, self.DIST_COEFF_FILENAME),D)

        self._camera_matrix = K
        self._distortion_coeffs = D

        return K, D, rvecs, tvecs

    def rectify(self, img, R=np.eye(3), newK=None):
        if newK is None:
            newK = self._camera_matrix

        if self._rectify_map1 is None:
            h,w = img.shape[:2]
            self._rectify_map1, self._rectify_map2 = cv2.fisheye.initUndistortRectifyMap(
                self._camera_matrix, self._distortion_coeffs, R, newK, (w,h), cv2.CV_16SC2
            )

        undistorted_img = cv2.remap(
            img,
            self._rectify_map1, self._rectify_map2,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        return undistorted_img


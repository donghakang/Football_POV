import numpy as np
import cv2
import glob

img_size = (6, 8)
img_h, img_w = img_size

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((img_h*img_w,3), np.float32)
objp[:,:2] = np.mgrid[0:img_h,0:img_w].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./data/gopro_*.png')

print('==== PRINT FILE NAME THAT IS NOT WORKING ====')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, img_size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, img_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(fname)


retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("================ret===================")
print(retval)
print("================mtx===================")
print(cameraMatrix)
print("================dist==================")
print(distCoeffs)
# print("================rvecs=================")
# print(rvecs)
# print("================tvecs=================")
# print(tvecs)

#
print('==== PRINT STATUS FOR OPTIMAL NEW CAMERA MATRIX ====')
for fname in images:
    im = cv2.imread(fname)
    h,  w = im.shape[:2]
    distCoeffs = np.array([-0.13615181, 0.53005398, 0, 0, 0])
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,(w,h),1,(w,h))

    if roi == (0,0,0,0):
        print(' X || ', fname)
    else:
        print(' O || ', fname)



# # undistort
im = cv2.imread('./data/gopro_calibration12.png')
h,  w = im.shape[:2]
distCoeffs = np.array([-0.13615181, 0.53005398, 0, 0, 0])
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,(w,h),1,(w,h))

if roi == (0,0,0,0):
    print(' X || ', fname)
else:
    print(' O || ', fname)

dst = cv2.undistort(im, cameraMatrix, distCoeffs, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./data/00calibresult.png',dst)


print('========================')

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(cameraMatrix,distCoeffs,None,newcameramtx,(w,h),5)
dst = cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./data/01calibresult.png',dst)


# #
cv2.destroyAllWindows()

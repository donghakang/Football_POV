import cv2
import numpy as np
import random
import glob


import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
from numpy.linalg import inv


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = .07


def alignImages(im1, im2):

  # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)


  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  imMatch_resize = cv2.resize(imMatches, (960, 540))
  cv2.imshow("match", imMatch_resize)
  cv2.waitKey(5000);



  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)


  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)


  # Use homography
  height, width = im2.shape
  # im1Reg = cv2.warpPerspective(im1, h, (width, height))       ## Error Occurs in the wapPerspective

  return h



def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Capture the video
    vidcap = cv2.VideoCapture('20200208_football_throw01.MP4')
    # vidcap = cv2.VideoCapture('20200208_football_throw02.MP4')
    success,image = vidcap.read()
    count = 0
    while success:
        if (count > 100 and count <= 160 and count % 3 == 0):
            cv2.imwrite("./data/frame%d.jpg" % int((count-100)/3), image)     # save frame as JPEG file
        success,image = vidcap.read()
        # print('Read a new frame: ', success, "======", count)
        count += 1


    for i in range (1, 20):
        im1 = cv2.imread('./data/frame%d.jpg' % int(i), 0)
        im2 = cv2.imread('./data/frame%d.jpg' % int(i+1), 0)
        h = alignImages(im1, im2)

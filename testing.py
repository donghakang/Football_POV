import cv2
import numpy as np
import matplotlib.pyplot as plt

import glob
import os

from despin_video.view_expansion.image_stitching import RotationAlignment

def main():
  folder = './outputs/20200208_football_throw02/debugs/rev1_056-064'
  imgfilelist = sorted([f for f in glob.glob(os.path.join(folder, '*.png'))])
  imglist = [cv2.imread(imf) for imf in imgfilelist[:2]]

  H,W = imglist[0].shape[:2]

  center = (int(W/2), int(H/2))
  rotater = RotationAlignment('')

  mask = cv2.circle(np.zeros((H,W),dtype='uint8'), center, min(list(center)), 1, -1)

  blend = rotater._alpha_blend(imglist[0], imglist[1], mask, 1)
  print(blend.dtype)
  print(np.max(blend))

  plt.figure()
  plt.imshow(mask, cmap='gray')

  plt.figure()
  plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))

  plt.show()

if __name__ == "__main__":
  main()
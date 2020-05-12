import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def distsq_to_line(m,b, x,y):
  xp = (x + m*y - m*b) / (1+m**2)
  yp = (m*x + (m**2)*y - (m**2)*b) / (1+m**2) + b
  return ((xp-x)**2 + (yp-y)**2)

def detect_up_angle(image, bordermask, imgcenter):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = image.astype('float') / 255
  H,W = image.shape[:2]

  padimg = cv2.copyMakeBorder(image, 1,1,1,1, cv2.BORDER_REPLICATE)
  Hp, Wp = padimg.shape[:2]

  clrgrad = np.zeros((Hp,Wp))
  # numpad orientation
  nbr7 = np.sum(np.square(padimg[:Hp-2, :Wp-2,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr8 = np.sum(np.square(padimg[:Hp-2, 1:Wp-1,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr9 = np.sum(np.square(padimg[:Hp-2, 2:Wp,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr4 = np.sum(np.square(padimg[1:Hp-1, :Wp-2,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr6 = np.sum(np.square(padimg[1:Hp-1, 2:Wp,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr1 = np.sum(np.square(padimg[2:Hp, :Wp-2,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr2 = np.sum(np.square(padimg[2:Hp, 1:Wp-1,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
  nbr3 = np.sum(np.square(padimg[2:Hp, 2:Wp,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)

  tmp = np.stack([nbr1, nbr2, nbr3, nbr4, nbr6, nbr7, nbr8, nbr9], axis=-1)
  clrgrad = np.max(tmp, axis=-1)

  dbg = clrgrad.copy()

  # for y in range(1,Hp-1):
  #   for x in range(1,Wp-1):
  #     nbrcomp = [np.linalg.norm((padimg[yy,xx,:]-padimg[y,x,:]).astype('float'))
  #       for yy in [y-1, y, y+1] for xx in [x-1, x, x+1]
  #     ]

  #     clrgrad[y,x] = max(nbrcomp)

  # clrgrad = clrgrad[1:Hp-1,1:Wp-1]
  nearborder = cv2.erode(bordermask, np.ones((9,9), np.uint8), iterations=2)
  nearborder = cv2.bitwise_not(nearborder)
  clrgrad[nearborder > 0] = 0.0

  # plt.figure()
  # plt.imshow(dbg, cmap='gray', vmin=0, vmax=np.max(clrgrad))
  # plt.axis('off')

  # plt.figure()
  # plt.imshow(dbg, cmap='jet', vmin=0, vmax=np.max(clrgrad))
  # plt.axis('off')

  # suppress smaller values
  n_pts = 10000
  thresh = np.partition(clrgrad, clrgrad.size-n_pts, axis=None)[-n_pts]

  clrgrad[clrgrad <= thresh] = 0.0
  clrgrad[clrgrad > thresh] = 1.0
  pt_r, pt_c = np.nonzero(clrgrad)
  n_pts = len(pt_r)

  iters = 1000
  best_slope = 0
  best_intercept = 0
  best_inliers = 0
  ransac_thresh = 10 #pixels away

  for i in range(iters):
    pts_idx = random.sample(range(n_pts), 2)
    pts = [(float(c),float(r)) for r,c in zip(pt_r[pts_idx], pt_c[pts_idx])]

    # form a line
    m = (pts[1][1]-pts[0][1]) / (pts[1][0]-pts[0][0]+1e-15)
    b = pts[1][1] - m * pts[1][0]

    dists = np.array([distsq_to_line(m,b,c,r) for r,c in zip(pt_r, pt_c)])
    inliers = np.count_nonzero(dists < ransac_thresh)

    if inliers > best_inliers:
      best_slope = m
      best_intercept = b
      best_inliers = inliers

  # get mean pixel intensity above and below the line
  x1 = 0
  y1 = best_intercept
  if y1 < 0:
    y1 = 0
    x1 = - best_intercept / best_slope

  x2 = W
  y2 = best_slope * W + best_intercept
  if y2 > H:
    y2 = H
    x2 = (H-best_intercept)/best_slope

  skymask = np.zeros((H,W))
  for x in range(int(x2)):
    y = int(best_slope*x + best_intercept)
    if y >=0 and y < H:
      skymask[:y, x] = 1
  for y in range(int(y2)):
    x = int((y-best_intercept) / best_slope)
    if x >= 0 and x < W:
      skymask[y, x:] = 1

  groundmask = cv2.bitwise_not(skymask)
  skymask = cv2.bitwise_and(skymask, skymask, mask=bordermask)
  groundmask = cv2.bitwise_and(groundmask, groundmask, mask=bordermask)

  aboveline = np.sum(gray[skymask > 0]) / np.count_nonzero(skymask)
  belowline = np.sum(gray[skymask == 0]) / np.count_nonzero(groundmask)
  upslope = 1.0/best_slope

  # 255 is white, 0 is black
  if aboveline < belowline:
    # skymask is wrong way, flip
    print('flipping sky')
    upslope = -upslope
    tmpmask = skymask.copy()
    skymask = groundmask.copy() #cv2.bitwise_not(skymask)
    groundmask = tmpmask

  deltax = imgcenter[0] #W/2.0
  deltay = (1.0/upslope) * deltax
  angle_from_vert = np.arctan2(deltay, deltax)

  dbgpt = np.array([
    [np.cos(angle_from_vert), -np.sin(angle_from_vert)],
    [np.sin(angle_from_vert), np.cos(angle_from_vert)]
  ]) @ np.array([[0, -imgcenter[1]]]).T

  angle_from_vert *= 180/np.pi
  horizon2center = np.sqrt(distsq_to_line(best_slope, best_intercept, imgcenter[0], imgcenter[1]))
  if skymask[int(imgcenter[1]),int(imgcenter[0])] > 0:
    # center of image is in the sky side, so assign negative distance
    horizon2center = -horizon2center

  print([x1,y1,x2,y2])
  print([aboveline, belowline])
  print([best_intercept, best_slope])
  print([angle_from_vert, horizon2center])

  # linepts = [(x, int(best_slope*x+best_intercept)) for x in range(int(x1),int(x2))]
  # dbg = cv2.cvtColor((clrgrad*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
  # for x,y in linepts:
  #   dbg = cv2.circle(dbg, (x,y), 5, (0,255,0), thickness=3)
  # dbg = cv2.line(dbg, (int(imgcenter[0]), int(imgcenter[1])), (int(dbgpt[0,0]+imgcenter[0]), int(dbgpt[1,0]+imgcenter[1])), (0,0,255), thickness=5)
  # plt.figure()
  # plt.subplot(1,2,1)
  # plt.imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
  # plt.subplot(1,2,2)
  # plt.imshow(skymask, cmap='gray')
  # plt.show()

  return angle_from_vert, horizon2center
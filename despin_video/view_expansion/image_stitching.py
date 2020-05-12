import inspect
import os
import shutil
import contextlib
import subprocess
import platform
import glob

import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

from collections import deque
from collections import OrderedDict

import despin_video.view_expansion.sky_detection as skydet

class ImageComposition:
  def __init__(self, folder_prefix, output_dir, backup_dir=None):
    '''
    @param backup_dir directory to load previously generated images from

    '''
    self.stitched_suffix = '-[NISwGSP][2D][BLEND_LINEAR]'
    self.folder_prefix = folder_prefix
    self.working_dir = os.getcwd()
    thisdir = os.path.dirname(os.path.realpath(__file__))
    self.niswgsp_dir = os.path.join(thisdir, 'niswgsp')
    self.input_42_dir = os.path.join(self.niswgsp_dir, 'input-42-data')
    self.results_dir = os.path.join(self.input_42_dir, '0_results')
    self.debugs_dir = os.path.join(self.input_42_dir, '1_debugs')

    self.raw_results_dir = os.path.join(output_dir, 'stitch_raw')
    self.my_results_dir = os.path.join(output_dir, 'stitch_clean')

    self.composite_names = []
    self.composite_centers = []
    self.composite_radii = []

  def calculate_centers(self):
    imgfilelist = sorted([f for f in glob.glob(os.path.join(self.raw_results_dir, '*.png'))])

    # complist = []
    self.composite_names = []
    self.composite_centers = []
    self.composite_radii = []

    for imfile in imgfilelist:
      print('calculating for ' + imfile)
      compname = os.path.splitext(os.path.basename(imfile))[0].replace(self.stitched_suffix, '')
      meshdir = os.path.join(self.debugs_dir, '{}-result'.format(compname))
      meshfile = os.path.join(meshdir, '{}{}[Mesh].png'.format(compname, self.stitched_suffix))

      comp, comp_center, comp_min_rad = self._find_rotation_center_with_mesh(imfile, meshfile)
      # comp = cv2.circle(comp.astype('uint8'), (int(comp_center[0]),int(comp_center[1])), int(comp_min_rad), (255,0,0), thickness=5)

      # complist.append(comp)
      self.composite_names.append(compname)
      self.composite_centers.append(comp_center)
      self.composite_radii.append(comp_min_rad)

  def retrieve_composites(self):
    '''
    This goes into the final results folder, and pulls out all of the composites we were able to create so far
    Returns a list of composite images
    '''
    imgfilelist = sorted([f for f in glob.glob(os.path.join(self.raw_results_dir, '*.png'))])

    # complist = []
    # compnames = []
    # comp_centers = []
    # comp_min_rads = []

    # for imfile in imgfilelist:
    #   print('retrieving ' + imfile)
    #   compname = os.path.splitext(os.path.basename(imfile))[0].replace(self.stitched_suffix, '')
    #   meshdir = os.path.join(self.debugs_dir, '{}-result'.format(compname))
    #   meshfile = os.path.join(meshdir, '{}{}[Mesh].png'.format(compname, self.stitched_suffix))

    #   comp, comp_center, comp_min_rad = self._find_rotation_center_with_mesh(imfile, meshfile)
    #   # comp = cv2.circle(comp.astype('uint8'), (int(comp_center[0]),int(comp_center[1])), int(comp_min_rad), (255,0,0), thickness=5)

    #   complist.append(comp)
    #   compnames.append(compname)
    #   comp_centers.append(comp_center)
    #   comp_min_rads.append(comp_min_rad)

    gminrad = int(min(self.composite_radii))
    print('min radius {}'.format(gminrad))

    croppeds = []
    for imfile, name, center in zip(imgfilelist, self.composite_names, self.composite_centers):
      # for comp, center in zip(complist, comp_centers):
      comp = cv2.imread(imfile, cv2.IMREAD_UNCHANGED)
      H,W = comp.shape[:2]
      mask = np.zeros((H,W))
      mask = cv2.circle(mask, (int(center[0]),int(center[1])), gminrad, 1, thickness=-1)

      leftx = int(center[0])-gminrad
      topy = int(center[1])-gminrad
      rightx = int(center[0])+gminrad
      bottomy = int(center[1])+gminrad

      if leftx < 0:
        leftx = 0
      if topy < 0:
        topy = 0
      if rightx > W:
        rightx = W
      if bottomy > H:
        bottomy = H

      comp = cv2.bitwise_and(comp, comp, mask=mask.astype('uint8'))
      cropped = comp[topy:bottomy, leftx:rightx, :]
      croppeds.append(cropped)

    return croppeds

  def postprocess(self):
    # Make sure the directory exists
    if os.path.isdir(self.my_results_dir):
      shutil.rmtree(self.my_results_dir)
    os.mkdir(self.my_results_dir)

    #####
    composites = self.retrieve_composites()
    for img, name in zip(composites, self.composite_names):
      cv2.imwrite(os.path.join(self.my_results_dir, '{}.png'.format(name)), img)

    return self.my_results_dir

  def compose(self, image_set, revolution_count, bottom_img):
    '''
    This is a wrapper interface around the C++ code
    accompanying the paper
    Natural Image Stitching with Global Similarity Prior

    Could rework it to be more seamless, but just use the code as it was designed to be.
    Write images to an output folder, then call the executable.

    NISwGSP expects the images to be in a folder at the same level as the executable called input-42:

    /niswgsp/
    |-- NISwGSP (executable)
    |-- input-42-data
        |-- 0_results
        |-- <my setname subdir>

    Then, the executable must be called at the same directory as such:
    ./NISwGSP <my setname subdir>

    The output images will be stored in 0_results

    2D_LINEAR seems to give the best results, so we'll grab those.
    '''
    assert platform.system() == 'Linux', 'This is only supported on Linux at the moment'
    assert len(image_set) > 1, 'not enough images!'

    ######
    success = False

    # form the subdirectories
    subdirname = self.folder_prefix + '_R{:02d}'.format(revolution_count)
    subdirpath = os.path.join(self.input_42_dir, subdirname)
    resdirpath = os.path.join(self.results_dir, '{}-result'.format(subdirname))

    # check if already exists. delete and remake if so
    if os.path.isdir(subdirpath):
      print('removing existing directory {}'.format(subdirpath))
      try:
        shutil.rmtree(subdirpath)
      except OSError as e:
        print('Failed to delete {} - {}'.format(subdirpath, e))
        print('skipping {}'.format(revolution_count))
        return None

    try:
      os.mkdir(subdirpath)
    except OSError as e:
      print("Creation of sub directory {} failed -- {}".format(subdirpath, e))

    # save all images to this subdir
    for im_num, img in enumerate(image_set, start=1):
      impath = os.path.join(subdirpath, '{:02d}.png'.format(im_num))
      cv2.imwrite(impath, img)

    # create a parameter file
    with open(os.path.join(subdirpath, '{}-STITCH-GRAPH.txt'.format(subdirname)), 'w+') as graphfile:
      # {center_image_index | 9 | center image index}
      # {center_image_rotation_angle | 0 | center image rotation angle}
      # {images_count | 14 | images count}
      # {matching_graph_image_edges-0 | 1 | matching graph image edge 0}
      nimages = len(image_set)
      graphlines = [
        '{{center_image_index | {:02d} | center image index}}\n'.format(bottom_img), # Just pick the middle image here..
        '{{center_image_rotation_angle | {} | center image rotation angle}}\n'.format(0),
        '{{images_count | {} | images count}}\n'.format(nimages)
      ] + [
        '{{matching_graph_image_edges-{} | {} | matching graph image edge {}}}\n'.format(im_num, (im_num+1)%nimages, im_num)
        for im_num in range(nimages-1)
      ]
      graphfile.writelines(graphlines)

    ### Now run the executable
    with contextlib.ExitStack() as stack:
      @stack.callback
      def return_to_working_dir():
        os.chdir(self.working_dir)
        print('returned to working directory')

      # change to NISwGSP directory
      os.chdir(self.niswgsp_dir)
      print('switching to NISwGSP directory')

      # call the executable
      resultproc = None
      logfile_path = os.path.join(self.results_dir, 'stdout.txt')
      with open(logfile_path, 'w+') as logfile:
        args = ('./NISwGSP_Ubuntu', subdirname)
        print('executing with args {}'.format(args))
        resultproc = subprocess.run(args, shell=False, cwd=self.niswgsp_dir, stdout=logfile, stderr=subprocess.STDOUT)

      if resultproc is None:
        print('{} File opening failed - moving on. '.format(subdirname))

      # move the log file to the output directory
      shutil.move(logfile_path, os.path.join(resdirpath, 'stdout.txt'))
      print('{} returned {} -- {}'.format(resultproc.args, resultproc.returncode, 'ERROR' if resultproc.returncode else 'SUCCESS'))

      if resultproc.returncode == 0:
        # copy the stitched image to combined result folder
        stitchedname = subdirname + self.stitched_suffix + '.png'
        shutil.copy2(os.path.join(resdirpath, stitchedname), os.path.join(self.raw_results_dir, '{}.png'.format(subdirname)))
        success = True

    return success

  def _find_rotation_center_with_mesh(self, compositefile, meshfile):
    comp = cv2.imread(compositefile, cv2.IMREAD_UNCHANGED)
    mesh = cv2.imread(meshfile, cv2.IMREAD_GRAYSCALE)

    clr = comp[:,:,:3]
    mask = comp[:,:,-1]

    gray = cv2.cvtColor(clr, cv2.COLOR_BGR2GRAY)
    meshdiff = mesh-gray
    meshdiff[meshdiff > 0] = 255

    H,W = meshdiff.shape
    pts = np.array([[c,r] for r in range(H) for c in range(W) if (meshdiff[r,c]>0)])
    center = np.mean(pts, axis=0)
    for i in range(20):
      neighbors = np.vstack([pt for pt in pts if np.linalg.norm(pt-center) < 80])
      center = np.mean(neighbors, axis=0)

    # find the min radius
    mask = mask.astype('float64') / 255.0
    sobelx = cv2.Sobel(mask, -1, 1, 0, ksize=1)
    sobely = cv2.Sobel(mask, -1, 0, 1, ksize=1)
    edge = np.sqrt(np.square(sobelx)+np.square(sobely))
    edge[edge > 0] = 1.0

    edgept_r, edgept_c = np.nonzero(edge)
    n_edges = len(edgept_r)

    radii = [np.sqrt((c-center[0])**2+(r-center[1])**2) for r,c in zip(edgept_r, edgept_c)]
    radii = [r for r in radii if r > 300]

    min_rad = min(radii)

    # plt.figure()
    # ax = plt.subplot(1,2,1)
    # dbg = gray.copy()
    # dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
    # dbg = cv2.circle(dbg, (int(center[0]), int(center[1])), 5, (0,255,0), thickness=5)
    # dbg = cv2.circle(dbg, (int(center[0]), int(center[1])), int(min_rad), (0,255,0), thickness=5)
    # plt.imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
    # ax.axis('off')
    # ax = plt.subplot(1,2,2)
    # plt.imshow(mesh, cmap='gray')
    # ax.axis('off')
    # plt.show()

    return (comp, center, min_rad)


class RotationAlignment:
  def __init__(self, output_dir):
    self.my_results_dir = os.path.join(output_dir, 'relative_rotation')

  def process(self, input_dir, imgcenters):
    # Clean out the directory
    if os.path.isdir(self.my_results_dir):
      shutil.rmtree(self.my_results_dir)
    os.mkdir(self.my_results_dir)

    #####
    imgfilelist = sorted([f for f in glob.glob(os.path.join(input_dir, '*.png'))])
    assert len(imgfilelist) > 0

    names = []
    imgs = []
    angles = []
    dists = []
    horizon_locations = []

    for imgf, center in zip(imgfilelist, imgcenters):
      name = os.path.splitext(os.path.basename(imgf))[0]
      img = cv2.imread(imgf, cv2.IMREAD_UNCHANGED)
      clr = img[:,:,:3]
      border = img[:,:,3]

      print('processing {}'.format(name))
      angle_from_vert_deg, dist2horizon = skydet.detect_up_angle(clr, border, center)

      names.append(name)
      imgs.append(img)
      angles.append(angle_from_vert_deg)
      dists.append(dist2horizon)
      horizon_locations.append(center[1] - dist2horizon)

    # horizon_locations = [center[1]-dist for center, dist in zip(imgcenters, dists)]
    new_horizon_height = np.mean(horizon_locations)
    print(dists)
    print(horizon_locations)
    print(new_horizon_height)

    newimgs = []
    newcenters = []
    proposed_szs = []
    for name, img, center, angle_from_vert_deg, y_horizon in zip(names, imgs, imgcenters, angles, horizon_locations):
      H,W = img.astype('int32').shape[:2]
      R = cv2.getRotationMatrix2D(tuple(center), angle_from_vert_deg, 1)
      aligned = cv2.warpAffine(img, R, (W,H))

      deltay = new_horizon_height - y_horizon

      newcenter = (center[0], center[1]) #(center[0], center[1]-deltay)
      print(center)
      print([deltay])
      print([H,W])
      print(newcenter)
      print(y_horizon)
      T = np.float32([[1,0,0],[0,1,deltay]])
      aligned2 = cv2.warpAffine(aligned, T, (W,H))

      # plt.figure()
      # plt.subplot(1,2,1)
      # plt.imshow(aligned)
      # plt.subplot(1,2,2)
      # plt.imshow(aligned2)
      # plt.show()

      # find tentative new boundaries to crop
      # use the middle third to find
      border = aligned2[:,:,3]
      x1 = 2*W/4.0
      x2 = 3*W/4.0
      y1 = 2*H/4.0
      y2 = 3*H/4.0
      top_vals = np.max(border[:,int(x1):int(x2)], axis=0)
      # if np.any(top_vals == 0):
      #   print('updating top_vals')
      #   x1 += np.argmax(top_vals)
      #   x2 -= np.argmax(top_vals[::-1])
      top_inds = np.argmax(border[:,int(x1):int(x2)], axis=0)
      bottom_inds = H - 1 - np.argmax(border[::-1,int(x1):int(x2)], axis=0)

      left_vals = np.max(border[int(y1):int(y2), :], axis=1)
      # if np.any(left_vals == 0):
      #   print('updating left vals')
      #   y1 += np.argmax(left_vals)
      #   y2 -= np.argmax(left_vals[::-1])

      left_inds = np.argmax(border[int(y1):int(y2), :], axis=1)
      right_inds = W - 1 - np.argmax(border[int(y1):int(y2), ::-1], axis=1)
      #
      top = np.max(top_inds)
      bottom = np.min(bottom_inds)
      left = np.max(left_inds)
      right = np.min(right_inds)

      print([x1,y1,x2,y2])
      # print(top_vals)
      # print(np.argmax(border[::-1,int(x1):int(x2)], axis=0))
      # print(left_vals)
      # print(np.argmax(border[int(y1):int(y2), ::-1], axis=1))

      newimgs.append(aligned2)
      newcenters.append(newcenter)
      proposed_szs.append([
        newcenter[1] - top,
        bottom - newcenter[1],
        newcenter[0] - left,
        right - newcenter[0]
      ])

    # pick best common size
    proposed_szs = np.array(proposed_szs)
    best_deltas = np.mean(proposed_szs, axis=0)
    deltatop = int(best_deltas[0])
    deltabottom = int(best_deltas[1])
    deltaleft = int(best_deltas[2])
    deltaright = int(best_deltas[3])
    for img, center in zip(newimgs, newcenters):
      H,W = img.shape[:2]
      leftx = int(center[0]) - deltaleft
      rightx = int(center[0]) + deltaright
      topy = int(center[1]) - deltatop
      bottomy = int(center[1]) + deltabottom
      if leftx < 0:
        deltaleft = deltaleft + leftx # leftx is negative, so use that to shrink
      if rightx > W:
        deltaright = deltaright - (rightx - W)
      if topy < 0:
        deltatop = deltatop + topy
      if bottomy > H:
        deltabottom = deltabottom - (bottomy - H)

    print(proposed_szs)
    print('best delta sizes: {}'.format(best_deltas))
    print('fixed: {}'.format([deltatop, deltabottom, deltaleft, deltaright]))
    new_w = deltaleft + deltaright
    new_h = deltatop + deltabottom

    vidout = cv2.VideoWriter(os.path.join(self.my_results_dir, 'vidout.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 2, (new_w, new_h))

    N_imgs = len(names)
    croppeds = []
    emptymasks = []
    for name, img, center in zip(names, newimgs, newcenters):
      H,W = img.shape[:2]
      leftx = int(center[0]) - deltaleft
      rightx = int(center[0]) + deltaright
      topy = int(center[1]) - deltatop
      bottomy = int(center[1]) + deltabottom

      if leftx < 0:
        print('warning - left side truncated ({})'.format(name))
        leftx = 0
      if rightx > W:
        print('warning - right side truncated ({})'.format(name))
        rightx = W
      if topy < 0:
        print('warning - top side truncated ({})'.format(name))
        topy = 0
      if bottomy > H:
        print('warning - bottom side truncated ({})'.format(name))
        bottomy = H

      crop = img[topy:bottomy, leftx:rightx, :3]
      mask = img[topy:bottomy, leftx:rightx, 3]
      emptymask = cv2.bitwise_not(mask)

      croppeds.append(crop)
      emptymasks.append(emptymask)

    for idx, name, crop, emptymask in zip(range(N_imgs), names, croppeds, emptymasks):
      emptymask = cv2.dilate(emptymask, np.ones((9,9), dtype=np.uint8))

      other_imgs = [j for j in range(N_imgs) if j != idx]
      filler = np.zeros_like(crop)
      alpha = 1.0
      for ith_other, j in enumerate(other_imgs):
        newmask = cv2.bitwise_and(emptymask, cv2.bitwise_not(emptymasks[j]))
        print([filler.shape, filler.dtype, np.max(filler)])
        print([croppeds[j].shape, croppeds[j].dtype, np.max(croppeds[j])])
        filler = self._alpha_blend(filler, croppeds[j], newmask, alpha)
        alpha = alpha * 0.5
        print('{}/{}: max={}'.format(idx, j, np.max(newmask)))
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(emptymask, cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(newmask, cmap='gray')
        # plt.suptitle('{} / {}'.format(idx, j))
        # plt.show()

      print([filler.dtype, filler.shape])

      result = crop.copy()
      # filler_2 = filler.astype('uint8')
      # for ch in range(3):
      #   tmp = result[:,:,ch]
      #   tmp_fill = filler_2[:,:,ch]
      #   tmp[np.where(emptymask > 0)] = tmp_fill[np.where(emptymask > 0)]

      # plt.figure()
      # plt.subplot(1,3,1)
      # plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
      # plt.subplot(1,3,2)
      # plt.imshow(filler)
      # plt.subplot(1,3,3)
      # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
      # plt.show()

      cv2.imwrite(os.path.join(self.my_results_dir, '{}.png'.format(name)), result)
      vidout.write(result)

    vidout.release()

    return self.my_results_dir

  def _alpha_blend(self, img1, img2, mask, alpha=0.5):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    assert alpha >= 0 and alpha <= 1
    blended = img1.copy()
    for ch in range(3):
      tmp = blended[:,:,ch]
      tmp1 = img1[:,:,ch]
      tmp2 = img2[:,:,ch]
      inds = np.where(mask > 0)
      tmp[inds] = tmp1[inds]*(1.0-alpha) + tmp2[inds]*alpha

    return blended

class CircleToBoxImage:
  def __init__(self, output_dir):
    self.ASPECT_RATIO = 16.0/10.0

    self.my_results_dir = os.path.join(output_dir, 'boxed')

  def crop(self, input_dir):
    # Clean out the directory
    if os.path.isdir(self.my_results_dir):
      shutil.rmtree(self.my_results_dir)
    os.mkdir(self.my_results_dir)

    #####
    imgfilelist = sorted([f for f in glob.glob(os.path.join(input_dir, '*.png'))])
    assert len(imgfilelist) > 0

    imglist = []
    imgnames = []
    for imgf in imgfilelist:
      name = os.path.splitext(os.path.basename(imgf))[0]
      imgnames.append(name)
      imglist.append(cv2.imread(imgf))

    centers = []
    rads = []
    for img in imglist:
      c,rad = self._ransac_circle(img)
      centers.append([int(c[0]), int(c[1])])
      rads.append(rad)

    common_rad = min(rads)
    corner_angle = np.arctan(1/self.ASPECT_RATIO)
    rectx_off = common_rad * np.cos(corner_angle)
    recty_off = rectx_off / self.ASPECT_RATIO

    # rectx_off = common_rad
    # recty_off = common_rad

    rectx_off = int(rectx_off)
    recty_off = int(recty_off)

    new_w = 2*rectx_off
    new_h = 2*recty_off

    new_imglist = []
    for img, c in zip(imglist, centers):
      H,W = img.shape[:2]

      leftx = c[0]-rectx_off
      topy = c[1]-recty_off
      rightx = c[0]+rectx_off
      bottomy = c[1]+recty_off

      if leftx < 0:
        leftx = 0
      if topy < 0:
        topy = 0
      if rightx > W:
        rightx = W
      if bottomy > H:
        bottomy = H

      cropped = img[topy:bottomy, leftx:rightx, :]
      new_imglist.append(cropped)

    # now save it off
    vidout = cv2.VideoWriter(os.path.join(self.my_results_dir, 'vidout.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 2, (new_w, new_h))
    for img, name in zip(new_imglist, imgnames):
      cv2.imwrite(os.path.join(self.my_results_dir, '{}.png'.format(name)), img)
      vidout.write(img)
    vidout.release()

    return self.my_results_dir

  def _define_circle(self, p1, p2, p3):
    """
    https://stackoverflow.com/a/50974391

    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

  def _ransac_circle(self, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype('float')
    sobelx = cv2.Sobel(mask, -1, 1, 0, ksize=1)
    sobely = cv2.Sobel(mask, -1, 0, 1, ksize=1)
    edge = np.sqrt(np.square(sobelx)+np.square(sobely))
    edge[edge > 0] = 1.0
    edgept_r, edgept_c = np.nonzero(edge)
    n_edges = len(edgept_r)

    # RANSAC
    iters = 1000
    best_center = (0,0)
    best_rad = 0
    best_inliers = 0

    for i in range(iters):
      pts_idx = random.sample(range(n_edges), 3)
      pts = [(c,r) for r,c in zip(edgept_r[pts_idx], edgept_c[pts_idx])]

      # for pt in pts:
      #   dbg = cv2.circle(dbg, pt, 10, (0,0,255), thickness=20)

      center, rad = self._define_circle(*pts)

      if center is None:
        continue

      # calculate number of inliers
      err = np.array([abs(np.sqrt((c-center[0])**2+(r-center[1])**2)-rad) for r,c in zip(edgept_r, edgept_c)])

      THRESH = 20 # px
      inliers = np.count_nonzero(err < THRESH) #len([p for p in err if (p < THRESH)])

      # print('min={}, max={}, median={}, #in={}'.format(min(err), max(err), np.median(err), inliers))

      # print('{}, {} = {} inliers'.format(center, rad, inliers))
      # dbg = cv2.circle(dbg, tuple(int(x) for x in center), int(rad), (255,0,0), thickness=4)

      if inliers > best_inliers:
        best_center = center
        best_rad = rad
        best_inliers = inliers

    return best_center, best_rad


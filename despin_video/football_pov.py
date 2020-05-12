import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

from despin_video.rectification.camera_calibration import FisheyeCalibration
from despin_video.view_expansion.triplet_generator import *
from despin_video.view_expansion.image_stitching import ImageComposition, CircleToBoxImage, RotationAlignment
from despin_video.interpolator.frame_interp import FrameInterpolation
from despin_video.horizontal_alignment.horizontal_alignment import *

class StabilizeFootballPOV:
  '''
  Main class for this project
  This is created as a context manager so that automatic cleanup happens on exit
  (see https://www.geeksforgeeks.org/context-manager-in-python/ )

  Usage:
    with StabilizeFootballPOV(vidfile) as fpov:
      fpov.run()

  '''
  STAGES = {
    stage: i
    for i,stage in enumerate([
      'STITCH',
      'POSTSTITCH',
      'PREROTATE',
      'ROTATE',
      'BOX',
      'INTERP',
      'DEBUG',
      'FINAL'
    ])
  }

  def __init__(self, raw_vid_file, outputsdir, start_step=0):
    self.raw_vid_file = raw_vid_file
    self.vidname = os.path.splitext(os.path.basename(raw_vid_file))[0]

    self.base_outputs_dir = outputsdir
    self.output_dir = os.path.join(self.base_outputs_dir, self.vidname)
    if not os.path.isdir(self.output_dir):
      os.mkdir(self.output_dir)

    assert start_step in self.STAGES.values(), 'Invalid start_step! choose from {}'.format(', '.join(['{}:{}'.format(i,stage) for stage,i in self.STAGES.items()]))
    self.current_step = start_step

    print('Processing {}'.format(self.vidname))

    ##### Create the components here, to get the output directories
    self._rectifier = FisheyeCalibration(load_coeffs=True)
    self._image_set_gen = Rotation_Imageset_Generator() #RevolutionFinder()
    self._composer = ImageComposition(self.vidname, self.output_dir, backup_dir=None)
    self._rotation_aligner = RotationAlignment(self.output_dir)
    self._horizon_aligner = HorizonAlignment(self.output_dir)
    self._boxer = CircleToBoxImage(self.output_dir)
    self._interp = FrameInterpolation(self.output_dir)
    #####

  def __enter__(self):
    self.rawcap = cv2.VideoCapture(self.raw_vid_file)
    assert self.rawcap.isOpened(), 'Error opening video file'

    # self.rawcap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    # VideoCapture Get method properties reference
    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    self.frame_width = int(self.rawcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.frame_height = int(self.rawcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.fps = self.rawcap.get(cv2.CAP_PROP_FPS)
    self.start_frame = int(self.rawcap.get(cv2.CAP_PROP_POS_FRAMES))

    print('FPS: {}'.format(self.fps))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # self.out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 40, (self.frame_width, self.frame_height))
    #self.out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 40.0, (self.frame_width, self.frame_height))

    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    # When everything done, release the video capture object
    self.rawcap.release()
    # self.out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

  def run(self):
    if self.current_step <= self.STAGES['STITCH']:
      print('========== Running stitching process ==========')
      self._run_stitch()

    if self.current_step <= self.STAGES['POSTSTITCH']:
      print('========== Running post-stitching ==========')
      self._composer.postprocess()

    next_folder = self._composer.raw_results_dir
    if self.current_step <= self.STAGES['PREROTATE']:
      print('========== Running rotational alignment ==========')
      if len(self._composer.composite_centers) == 0:
        self._composer.calculate_centers()
      self._rotation_aligner.process(next_folder, self._composer.composite_centers)

    # next_folder = self._composer.my_results_dir
    # if self.current_step <= self.STAGES['ROTATE']:
    #   print('========== Running horizon alignment ==========')
    #   self._horizon_aligner.align_cnn(next_folder, crop=False)

    # next_folder = self._horizon_aligner.my_results_dir
    # if self.current_step <= self.STAGES['BOX']:
    #   print('========== Running box cropping ==========')
    #   self._boxer.crop(next_folder)

    next_folder = self._rotation_aligner.my_results_dir
    if self.current_step <= self.STAGES['INTERP']:
      print('========== Running interpolation ==========')
      self._interp.process(next_folder)

    if self.current_step <= self.STAGES['DEBUG']:
      print('========== Running debug ===========')
      self._run_debug()

    if self.current_step <= self.STAGES['FINAL']:
      print('========== Finalizing ==========')
      plt.show()

  def _run_stitch(self):
    # This is a special case, since compose() is per revolution
    # Clean out the directory
    if os.path.isdir(self._composer.raw_results_dir):
      shutil.rmtree(self._composer.raw_results_dir)
    os.mkdir(self._composer.raw_results_dir)

    ########
    frame_count = int(self.rawcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = self.start_frame
    revolution_counter = 0

    # Read until video is completed!
    while(self.rawcap.isOpened()):
      # Capture frame-by-frame
      ret, frame_bgr = self.rawcap.read()

      if ret == True:
        # 2.1 Remove Camera Distortion
        # I'm leaving this step outside of the 'kept images' in case undistorting the images affects
        # the mean pixel intensity comparison.  It shouldn't.
        # https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
        frame_bgr = self._rectifier.rectify(frame_bgr)

        frame_bgr = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # 2.2 View Expansion
        # We do this by first generating small composite images using sets of
        # 3 temporally neighboring images. Next, we further expand
        # the triplet image by using neighboring triplet images from
        # adjacent rotation cycles
        if self._image_set_gen.process_frame(frame_bgr, frame_number):
          rev_imgs = self._image_set_gen.last_image_set
          revolution_counter += 1
          print('Processing {}-frame revolution at frame {} -- {} s/rev = {} Hz rev'.format(len(rev_imgs), frame_number, len(rev_imgs)/float(self.fps), self.fps/float(len(rev_imgs))))
          self._composer.compose(rev_imgs, revolution_counter, self._image_set_gen.last_valley_idx)

        # Display the resulting frame
        cv2.imshow('Frame',frame_bgr)

        # Press Q on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
          break

        frame_number += 1

      # Break the loop
      else:
        break

    # DEBUG OUTPUTS
    plt.figure()
    plt.plot(range(len(self._image_set_gen.mean_intensities)), self._image_set_gen.mean_intensities, 'o-')
    # plt.plot(self._image_set_gen._movavg_list, '.-')
    # for rev_s, rev_e in self._image_set_gen._rev_pts:
    #   plt.plot([rev_s, rev_e], [self._image_set_gen.mean_intensities[rev_s], self._image_set_gen.mean_intensities[rev_e]], '^-')
    for rev_frames in self._image_set_gen.image_set_indexes:
      mi_vals = [self._image_set_gen.mean_intensities[fnum] for fnum in rev_frames]
      plt.plot(rev_frames, mi_vals, '^-', markersize=15)
    plt.grid()
    plt.xlabel('Frame')
    plt.ylabel('Mean Pixel Intensity')

    plt.figure()
    plt.plot(self._image_set_gen._sum_gradients, '.-')
    plt.grid()

    ## Clean up the composite images for post-processing
    self._composer.calculate_centers()
    return self._composer.raw_results_dir

  def _run_debug(self):
    if not self._image_set_gen._first_rev:
      print('already processed video. skipping debug')
      return

    # This is a special case, since compose() is per revolution
    # Clean out the directory
    debug_dir = os.path.join(self.output_dir, 'debugs')
    if os.path.isdir(debug_dir):
      shutil.rmtree(debug_dir)
    os.mkdir(debug_dir)

    frame_count = int(self.rawcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = self.start_frame
    revolution_counter = 0

    # Read until video is completed!
    while(self.rawcap.isOpened()):
      # Capture frame-by-frame
      ret, frame_bgr = self.rawcap.read()

      if ret == True:
        # 2.1 Remove Camera Distortion
        # I'm leaving this step outside of the 'kept images' in case undistorting the images affects
        # the mean pixel intensity comparison.  It shouldn't.
        # https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
        frame_bgr = self._rectifier.rectify(frame_bgr)

        frame_bgr = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # 2.2 View Expansion
        # We do this by first generating small composite images using sets of
        # 3 temporally neighboring images. Next, we further expand
        # the triplet image by using neighboring triplet iqmages from
        # adjacent rotation cycles
        if self._image_set_gen.process_frame(frame_bgr, frame_number):
          rev_imgs = self._image_set_gen.last_image_set
          revolution_counter += 1

          # rev_dir = os.path.join(debug_dir, 'rev{:01d}_{:03d}-{:03d}'.format(revolution_counter, self._image_set_gen.last_peak1_idx+self._image_set_gen._frame_offset, self._image_set_gen.last_peak2_idx+self._image_set_gen._frame_offset))
          rev_dir = os.path.join(debug_dir, 'rev{:01d}_{:03d}-{:03d}'.format(revolution_counter, self._image_set_gen.image_set_indexes[-1][0], self._image_set_gen.image_set_indexes[-1][-1]))
          os.mkdir(rev_dir)
          print('saving revolution to {}'.format(rev_dir))
          for idx, img in enumerate(rev_imgs):
            cv2.imwrite(os.path.join(rev_dir, '{:02d}.png'.format(idx)), img)

        # Display the resulting frame
        cv2.imshow('Frame',frame_bgr)

        # Press Q on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
          break

        frame_number += 1

      # Break the loop
      else:
        break

    # DEBUG OUTPUTS
    plt.figure()
    plt.plot(range(len(self._image_set_gen.mean_intensities)), self._image_set_gen.mean_intensities, 'o-')
    # plt.plot(self._image_set_gen._movavg_list, '.-')
    # for rev_s, rev_e in self._image_set_gen._rev_pts:
    #   plt.plot([rev_s, rev_e], [self._image_set_gen.mean_intensities[rev_s], self._image_set_gen.mean_intensities[rev_e]], '^-')
    for rev_frames in self._image_set_gen.image_set_indexes:
      mi_vals = [self._image_set_gen.mean_intensities[fnum] for fnum in rev_frames]
      plt.plot(rev_frames, mi_vals, '^-', markersize=15)
    plt.grid()
    plt.xlabel('Frame')
    plt.ylabel('Mean Pixel Intensity')

    plt.figure()
    plt.plot(self._image_set_gen._sum_gradients, '.-')
    plt.grid()




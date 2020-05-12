import os
import glob
import shutil
import subprocess

import cv2
import numpy as np

class FrameInterpolation:
  def __init__(self, output_dir):
    self.my_results_dir = os.path.join(output_dir, 'video')

    thisdir = os.path.dirname(os.path.realpath(__file__))
    self._sepconv_dir = os.path.join(thisdir, 'sepconv-slomo')
    self._exec_script = os.path.join(self._sepconv_dir, 'run.py')

  def _interpolate2x(self, directory, firstnum, secondnum):
    middlenum = (firstnum + secondnum)/2.0
    print('interpolating {} and {}'.format(firstnum, secondnum))
    firstfile = os.path.join(directory, '{:04d}.png'.format(int(firstnum * 100)))
    secondfile = os.path.join(directory, '{:04d}.png'.format(int(secondnum * 100)))
    middlefile = os.path.join(directory, '{:04d}.png'.format(int(middlenum * 100)))

    execcmd = 'python3 {} --model lf --first {} --second {} --out {}'.format(self._exec_script, firstfile, secondfile, middlefile)
    resultproc = subprocess.run(execcmd, shell=True, cwd=self._sepconv_dir)
    if resultproc.returncode != 0:
      print('Error {} running "{}"'.format(resultproc.returncode, execcmd))
      print('stdout: {}'.format(resultproc.stdout))
      print('stderr: {}'.format(resultproc.stderr))

  def process(self, input_dir):
    # Clean out the directory
    if os.path.isdir(self.my_results_dir):
      shutil.rmtree(self.my_results_dir)
    os.mkdir(self.my_results_dir)

    #####
    imgfilelist = sorted([f for f in glob.glob(os.path.join(input_dir, '*.png'))])
    assert len(imgfilelist) > 0

    # copy the first image
    shutil.copyfile(imgfilelist[0], os.path.join(self.my_results_dir, '0100.png'))

    # copy the images to the output folder enumerated
    for i, imgf in enumerate(imgfilelist[1:], start=2):
      # copy to output folder
      shutil.copyfile(imgf, os.path.join(self.my_results_dir, '{:02d}00.png'.format(i)))

      # run for half
      prev = i-1
      half = i-0.5
      self._interpolate2x(self.my_results_dir, prev, i)
      self._interpolate2x(self.my_results_dir, prev, half)
      self._interpolate2x(self.my_results_dir, half, i)

    # gather all the frames and make a video
    new_img_files = sorted([f for f in glob.glob(os.path.join(self.my_results_dir, '*.png'))])
    imgs = [cv2.imread(imgf) for imgf in new_img_files]
    vidout = cv2.VideoWriter(os.path.join(self.my_results_dir, 'vidout.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
      vidout.write(img)
    vidout.release()

    return self.my_results_dir




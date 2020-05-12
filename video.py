import os
import sys
import argparse

from despin_video.football_pov import StabilizeFootballPOV

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('vidpath', help='path to video file for processing')
  parser.add_argument('--skipto', type=int, choices=StabilizeFootballPOV.STAGES.values(), default=0,
                    help='Skip to steps in processing, {}'.format(', '.join(['{}:{}'.format(i,stage) for stage,i in StabilizeFootballPOV.STAGES.items()])))

  args = parser.parse_args()

  thisdir = os.path.dirname(os.path.realpath(__file__))

  with StabilizeFootballPOV(args.vidpath, os.path.join(thisdir, 'outputs'), args.skipto) as football:
    football.run()




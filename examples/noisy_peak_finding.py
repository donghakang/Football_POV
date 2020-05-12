import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class RevolutionFinder:
  def __init__(self, *args):
    self._movavg_queue = deque(maxlen=5)

    self._peak1 = 0
    self._peak1_ind = 0
    self._peak2 = 0
    self._peak2_ind = 0
    self._valley1 = np.inf
    self._valley1_ind = 0
    self._skip_ctr = 0

    self.last_image_set = []

    self._movavg_list = []

  def process_frame(self, frame, frame_number):
    image_set_generated = False
    self._movavg_queue.append(frame)
    proc_ind = frame_number - 2 # delay due to moving average

    if proc_ind >= 0:
      movavg = sum(self._movavg_queue) / float(self._movavg_queue.maxlen)
      self._movavg_list.append(movavg)

      # search for a peak-valley-peak
      if self._peak2_ind == 0 and movavg > self._peak1:
        self._peak1_ind = proc_ind
        self._peak1 = movavg
      elif self._peak1_ind > 0 and movavg < self._valley1:
        self._valley1_ind = proc_ind
        self._valley1 = movavg
      elif self._valley1_ind > 0 and movavg > self._peak2:
        self._peak2_ind = proc_ind
        self._peak2 = movavg
      elif self._skip_ctr < 2:
        # do this one more time to make sure it wasn't a false alarm.
        self._skip_ctr += 1
      else:
        # found candidate revolution. test and reset
        nframes = self._peak2_ind - self._peak1_ind
        print('{}-frame candidate: frame {} - {}, {}'.format(nframes, self._peak1_ind, self._peak2_ind, self._peak1-self._valley1))
        if self._peak1-self._valley1 > 20 and nframes < 17 and nframes > 4:
          image_set_generated = True
          self.last_image_set = (self._peak1_ind, self._peak2_ind)

        self._peak1 = self._peak2
        self._peak1_ind = self._peak2_ind+1
        self._peak2 = 0
        self._peak2_ind = 0
        self._valley1 = np.inf
        self._valley1_ind = 0
        self._skip_ctr = 0

    return image_set_generated

def main(mifi):
  mi_list = np.load(mifi)
  N = len(mi_list)
  movavg_queue = deque(maxlen=5)
  movavg = np.zeros_like(mi_list)

  search_queue = []
  peak1 = 0
  peak1_ind = 0
  peak2 = 0
  peak2_ind = 0
  valley1 = np.inf
  valley1_ind = 0
  skip_ctr = 0

  revs = []
  finder = RevolutionFinder()

  for i, mi in enumerate(mi_list):
    if finder.process_frame(mi, i):
      revs.append(finder.last_image_set)

    # search_queue.append(mi)
    # movavg_queue.append(mi)
    # proc_ind = i-2
    # if proc_ind >= 0:
    #   movavg[proc_ind] = sum(movavg_queue) / float(movavg_queue.maxlen)

    #   # search for a peak-valley-peak
    #   if peak2_ind == 0 and movavg[proc_ind] > peak1:
    #     peak1_ind = proc_ind
    #     peak1 = movavg[proc_ind]
    #   elif peak1_ind > 0 and movavg[proc_ind] < valley1:
    #     valley1_ind = proc_ind
    #     valley1 = movavg[proc_ind]
    #   elif valley1_ind > 0 and movavg[proc_ind] > peak2:
    #     peak2_ind = proc_ind
    #     peak2 = movavg[proc_ind]
    #   elif skip_ctr < 2:
    #     # do this one more time to make sure it wasn't a false alarm.
    #     skip_ctr += 1
    #   else:
    #     # found candidate revolution. test and reset
    #     nframes = peak2_ind - peak1_ind
    #     print('{}-frame candidate: frame {} - {}, {}'.format(nframes, peak1_ind, peak2_ind, peak1-valley1))
    #     if peak1-valley1 > 20 and nframes < 17 and nframes > 4:
    #       print('Found')
    #       revs.append((peak1_ind, peak2_ind))

    #     peak1 = peak2
    #     peak1_ind = peak2_ind+1
    #     peak2 = 0
    #     peak2_ind = 0
    #     valley1 = np.inf
    #     valley1_ind = 0
    #     skip_ctr = 0

  plt.figure()
  plt.plot(mi_list, '.-')
  plt.plot(finder._movavg_list, '.-')
  for rev_s, rev_e in revs:
    plt.plot([rev_s, rev_e], [mi_list[rev_s], mi_list[rev_e]], 'o-')
  plt.grid()
  plt.show()


if __name__ == "__main__":
  mifiles = [
    '20200208_football_throw02_mean_intensities.npy'
  ]

  main(mifiles[0])

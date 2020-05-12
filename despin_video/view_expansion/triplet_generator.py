import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt

class TripletGenerator:
    def __init__(self, max_frames, frame_height, frame_width, valley_frame_distance):
        # set constructor values
        self._max_frames = max_frames

        self._queue_len = (((valley_frame_distance * 2) + 1) + 6) # how many we're keeping on each sidex2, plus the middle, plus 3 on each side
        self._rolling_frame_intensity_list = deque(maxlen=self._queue_len)
        self._rolling_frame_image_list = deque(maxlen=self._queue_len) # look at x images at a time for valley detection
        self._rolling_frame_number_list = deque(maxlen=self._queue_len) # track corresponding x frame numbers at a time for valley detection
        self._previous_max = 0
        # # distance from the valley to keep
        # self._valley_frame_distance = valley_frame_distance
        self._center_index = int(self._queue_len / 2)
        self._queue_indices = [qi for qi in range(self._center_index - valley_frame_distance, self._center_index + valley_frame_distance+1)]

        # debug
        self._manual_frame_offset = 2
        self._triplets = []
        self._rolling_frame_number_list = deque(maxlen=self._queue_len) # track corresponding x frame numbers at a time for valley detection
        self._triplet_indexes  = [] # for debugging only
        self.mean_intensities = np.zeros((max_frames,))

    def get_triplet_indexes(self):
        return self._triplet_indexes

    # returns full list of triplets gathered from the video.
    # we could refactor to  eliminate a full list if we need
    # to for performance, once we're comfortable.
    def get_triplets(self):
        return self._triplets

    # returns last triplet
    def get_last_triplet(self):
        triplet = [self._rolling_frame_image_list[idx + self._manual_frame_offset] for idx in self._queue_indices]
        self._triplets.append(triplet)
        self._triplet_indexes.append([self._rolling_frame_number_list[idx + self._manual_frame_offset] for idx in self._queue_indices])
        return triplet

    def process_frame(self, frame_rgb, frame_number):
        # keep full list of frame intensities, so that we can plot it
        # convert to grayscale
        frame = cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2GRAY)

        mean_intensity = np.mean(frame)

        if mean_intensity > self._previous_max:
            self._previous_max = mean_intensity

        # stack the new frame to the end of the rolling list
        self._rolling_frame_image_list.append(frame_rgb)
        self._rolling_frame_number_list.append(frame_number)
        self._rolling_frame_intensity_list.append(mean_intensity)
        self.mean_intensities[frame_number] = mean_intensity

        # need to wait for queue to fill up first
        if len(self._rolling_frame_intensity_list) != self._queue_len:
            return False

        # The intensity list is the FULL intensity list for all frames, so we need to offset from the tail end.
        # So, the 5 items we're checking are at offset indexes -4,-3,-2, -1, 0
        # So, we're checking if -2 is less than the other surrounding 4.... in other words, is it the bottom of a valley?
        center_intensity = self._rolling_frame_intensity_list[self._center_index]
        intensity_comparisons = [intensity > center_intensity for idx, intensity in enumerate(self._rolling_frame_intensity_list) if idx != self._center_index]
        trip_generated = all(intensity_comparisons) and (self._previous_max-center_intensity) > 50
        trip_generated = trip_generated or (frame_number == 90)

        if trip_generated:
            # reset previous max to current valley
            self._previous_max = center_intensity
            # save the triplet frame indexes, so that we could use for debug purposes if we wanted.
            #self._triplet_indexes.append([frame_number-(2+self._valley_frame_distance), frame_number-2, frame_number-(2-self._valley_frame_distance)])

        return trip_generated

class Rotation_Imageset_Generator:
    def __init__(self):

        self._rolling_frame_image_list = []
        self._rolling_frame_number_list = []
        self._rolling_frame_mi_list = []

        self._previous_max = 0
        self._first_peak_found = False
        self._first_peak_mi = 0
        self._first_peak_list_index = 0
        self._second_peak_mi = 0
        self._second_peak_list_index = 0

        self._last_mi_range = 100 * 2.0

        self.last_image_set = []
        self.last_valley_idx = 0
        self._first_rev = True
        self._gradients_list = []

        self._fast = cv2.FastFeatureDetector_create()

        # debug
        self.image_sets = []
        self.image_set_indexes  = []
        self.mean_intensities = []
        self._sum_gradients = []

    def process_frame(self, frame_rgb, frame_number):
        # keep full list of frame intensities, so that we can plot it
        # convert to grayscale
        frame = cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2GRAY)

        mean_intensity = np.mean(frame)
        self.mean_intensities.append(mean_intensity)

        # padimg = cv2.copyMakeBorder(frame_rgb, 1,1,1,1, cv2.BORDER_REPLICATE)
        # Hp,Wp = padimg.shape[:2]
        # clrgrad = np.zeros((Hp,Wp))
        # # numpad orientation
        # nbr7 = np.sum(np.square(padimg[:Hp-2, :Wp-2,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr8 = np.sum(np.square(padimg[:Hp-2, 1:Wp-1,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr9 = np.sum(np.square(padimg[:Hp-2, 2:Wp,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr4 = np.sum(np.square(padimg[1:Hp-1, :Wp-2,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr6 = np.sum(np.square(padimg[1:Hp-1, 2:Wp,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr1 = np.sum(np.square(padimg[2:Hp, :Wp-2,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr2 = np.sum(np.square(padimg[2:Hp, 1:Wp-1,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)
        # nbr3 = np.sum(np.square(padimg[2:Hp, 2:Wp,:] - padimg[1:Hp-1, 1:Wp-1,:]), axis=2)

        # tmp = np.stack([nbr1, nbr2, nbr3, nbr4, nbr6, nbr7, nbr8, nbr9], axis=-1)
        # clrgrad = np.max(tmp, axis=-1)
        # maxclrdist = np.sqrt(np.max(clrgrad))

        kp = self._fast.detect(frame, None)

        self._sum_gradients.append(len(kp))
        self._gradients_list.append(len(kp))

        self._rolling_frame_image_list.append(frame_rgb)
        self._rolling_frame_number_list.append(frame_number)
        self._rolling_frame_mi_list.append(mean_intensity)

        if not self._first_peak_found:
            if mean_intensity > self._first_peak_mi:
                self._first_peak_mi = mean_intensity
                self._first_peak_list_index = len(self._rolling_frame_image_list) -1
        else:
            if mean_intensity > self._second_peak_mi:
                self._second_peak_mi = mean_intensity
                self._second_peak_list_index = len(self._rolling_frame_image_list) -1

        image_set_generated = False
        # if we haven't found the first peak yet, we'lre looking for the drop from the mean intensity
        # currently marked as the first peak.
        if not self._first_peak_found and self._first_peak_mi-mean_intensity > 50:
            self._first_peak_found = True
            # reset the lists, so the peak frame is the first in the list.
            # self._rolling_frame_image_list = self._rolling_frame_image_list[self._first_peak_list_index:]
            # self._rolling_frame_number_list = self._rolling_frame_number_list[self._first_peak_list_index:]
            # self._rolling_frame_mi_list = self._rolling_frame_mi_list[self._first_peak_list_index:]

        # if we already found the first peak, then we're looking for the drop from the mean intensity
        # currently marked as the second peak.  When we've found this, we have a full rotation
        elif self._first_peak_found and self._second_peak_mi-mean_intensity > 50 and (self._second_peak_list_index - self._first_peak_list_index) > 3:
            self._first_rev = False

            image_set_generated = True
            # save off the rotation
            img_set = self._rolling_frame_image_list[self._first_peak_list_index:self._second_peak_list_index]
            fnum_set = self._rolling_frame_number_list[self._first_peak_list_index:self._second_peak_list_index]
            mi_set = self._rolling_frame_mi_list[self._first_peak_list_index:self._second_peak_list_index]
            grad_set = self._gradients_list[self._first_peak_list_index:self._second_peak_list_index]

            self.last_image_set = []
            tmp_fnum_set = []
            tmp_mi_set = []
            for img, fnum, mi, grad in zip(img_set, fnum_set, mi_set, grad_set):
                if grad > 300:
                    self.last_image_set.append(img)
                    tmp_fnum_set.append(fnum)
                    tmp_mi_set.append(mi)

            if len(self.last_image_set) < len(img_set):
                print('truncating revolution frames from {} to {} frames'.format(len(img_set), len(self.last_image_set)))

            self.image_sets.append(self.last_image_set)
            self.image_set_indexes.append(tmp_fnum_set)

            last_min_mi_idx = np.argmin(tmp_mi_set)
            self.last_valley_idx = last_min_mi_idx

            # self._last_mi_range = self._first_peak_mi - self._rolling_frame_mi_list[last_min_mi_idx]

            # print('updating mean intensity range to {}'.format(self._last_mi_range))

            # reset rolling lists
            self._rolling_frame_image_list = self._rolling_frame_image_list[self._second_peak_list_index:]
            self._rolling_frame_number_list = self._rolling_frame_number_list[self._second_peak_list_index:]
            self._rolling_frame_mi_list = self._rolling_frame_mi_list[self._second_peak_list_index:]
            self._gradients_list = self._gradients_list[self._second_peak_list_index:]

            # reset first and second peak values
            self._first_peak_mi = self._second_peak_mi
            self._first_peak_list_index = 0
            self._second_peak_mi = mean_intensity
            self._second_peak_list_index = len(self._rolling_frame_image_list) -1

        return image_set_generated

class RevolutionFinder:
    def __init__(self, *args):
        self._movavg_queue = deque(maxlen=5)

        self._first_rev = True
        self._peak1 = 0
        self._peak1_ind = None
        self._peak2 = 0
        self._peak2_ind = None
        self._valley1 = np.inf
        self._valley1_ind = None
        self._skip_ctr = 0
        self._frame_offset = 0
        self._last_movavg = 0

        self.last_image_set = []
        self.last_peak1_idx = 0
        self.last_peak2_idx = 0
        self.last_valley_idx = 0
        self._image_list = []

        self._movavg_list = []
        self.mean_intensities = []
        self._rev_pts = []

        self._gradients_list = []
        self._sum_gradients = []

    def process_frame(self, frame_rgb, frame_number):
        image_set_generated = False

        frame = cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(frame)

        padimg = cv2.copyMakeBorder(frame_rgb, 1,1,1,1, cv2.BORDER_REPLICATE)
        Hp,Wp = padimg.shape[:2]
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
        maxclrdist = np.sqrt(np.max(clrgrad))
        # sobelx = cv2.Sobel(frame.astype('float'), -1, 1, 0, ksize=1)
        # sobely = cv2.Sobel(frame.astype('float'), -1, 0, 1, ksize=1)
        # sumgrad = np.sum(np.sqrt(np.square(sobelx)+np.square(sobely)))
        self._sum_gradients.append(maxclrdist)

        self.mean_intensities.append(mean_intensity)
        self._image_list.append(frame_rgb)
        self._gradients_list.append(maxclrdist)
        self._movavg_queue.append(mean_intensity)
        proc_ind = frame_number-2 - self._frame_offset # delay due to moving average

        if proc_ind >= 0:
            movavg = sum(self._movavg_queue) / float(self._movavg_queue.maxlen)
            self._movavg_list.append(movavg)

            # search for a peak-valley-peak
            if self._valley1_ind is None and movavg > self._peak1:
                self._peak1_ind = proc_ind
                self._peak1 = movavg
            elif self._valley1_ind is None and self._peak1_ind is not None and abs(self._last_movavg-movavg) <= 1:
                self._peak1_ind = proc_ind
                self._peak1 = movavg
            elif self._peak2_ind is None and movavg < self._valley1:
                self._valley1_ind = proc_ind
                self._valley1 = movavg
            elif self._valley1_ind is not None and movavg > self._peak2:
                self._peak2_ind = proc_ind
                self._peak2 = movavg
            elif self._skip_ctr < 4:
                # do this to make sure it wasn't a false alarm.
                self._skip_ctr += 1
            else:
                # found candidate revolution. test and reset
                nframes = self._peak2_ind - self._peak1_ind
                print('{}-frame candidate: frame {}-{}-{}, {}'.format(nframes, self._peak1_ind+self._frame_offset, self._valley1_ind+self._frame_offset, self._peak2_ind+self._frame_offset, self._peak1-self._valley1))
                if (self._peak1-self._valley1 > 5) and (self._peak2-self._valley1 > 5) and nframes < 30 and nframes > 4:
                    if self._first_rev:
                        self._peak1_ind += 1
                        self._first_rev = False

                    # image_set_generated = True
                    print('Found')

                    grad_set = self._gradients_list[self._peak1_ind:self._peak2_ind]
                    # print(grad_set)
                    img_set = self._image_list[self._peak1_ind:self._peak2_ind]
                    pk1ind = self._peak1_ind+self._frame_offset
                    # endind = self._peak2_ind #next((i for i,g in enumerate(grad_set) if g < 150000), self._peak2_ind)
                    self.last_image_set = [img for img, grad in zip(img_set, grad_set) if grad > 22]
                    # self.last_image_set = img_set[:endind]
                    self._rev_pts.append((pk1ind, pk1ind+len(self.last_image_set)))

                    self.last_peak1_idx = 0
                    self.last_peak2_idx = self._peak2_ind - self._peak1_ind
                    self.last_valley_idx = self._valley1_ind - self._peak1_ind
                    if self.last_valley_idx > len(self.last_image_set):
                        self.last_valley_idx = int(len(self.last_image_set)/2)

                    new_nframes = len(self.last_image_set)
                    if new_nframes > 1:
                        image_set_generated = True
                        if new_nframes == nframes:
                            print('Triggered')
                        else:
                            print('Triggered, reduced to {} frames'.format(new_nframes))

                self._gradients_list = self._gradients_list[self._peak2_ind:]
                self._image_list = self._image_list[self._peak2_ind:]
                self._frame_offset += self._peak2_ind
                self._peak1 = self._peak2
                self._peak1_ind = 0
                self._peak2 = 0
                self._peak2_ind = None
                self._valley1 = np.inf
                self._valley1_ind = None
                self._skip_ctr = 0

            self._last_movavg = movavg

        return image_set_generated
from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import listdir, sep
from os.path import isfile, join
import glob



class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


class Pipeline(object):
    def __init__(self):
        pass

    def run(self, image=None, debug=False):

        # get camera calibration parameters
        mtx,dist = camera_calibration().calibration_params()

        if debug:
            # inspect calibration
            test_images = [cv2.imread(i) for i in glob.glob('./test_images/*.jpg')]
            test_calibration(test_images[0], mtx,dist)

        # Undistort image
        undistorted = undistort(image, mtx=mtx, dist=dist)

        if debug:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            f.subplots_adjust(hspace=.2, wspace=.05)
            ax1.imshow(image)
            ax1.set_title('Original', fontsize=30)
            ax2.imshow(undistorted)
            ax2.set_title('Undistorted', fontsize=30)
            plt.imshow(undistorted)
            plt.show()

            visualize_color_channels(image)
            abs_sobel = abs_sobel_thresh(image, thresh=(25, 255))
            mag_sobel = mag_thresh(image, thresh=(25, 255))
            dir_sobel = dir_threshold(image, sobel_kernel=7, thresh=(0.8, np.pi/2)) #np.pi/2))
            hls_s = hls_s_select(image, thresh=(25, 255))
            sobel_combine_ = sobel_combine(image, s_thresh=(170, 255), sx_thresh=(20, 100))

            combined = np.zeros_like(mag_sobel)
            combined[((mag_sobel == 1) & (dir_sobel == 1))] = 1

            fig, axs = plt.subplots(3, 3, figsize=(16, 12))
            fig.subplots_adjust(hspace=.2, wspace=.001)
            axs = axs.ravel()
            #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].imshow(abs_sobel, cmap="gray")
            axs[0].set_title('Abs sobel', fontsize=30)
            axs[1].imshow(mag_sobel, cmap='gray')
            axs[1].set_title('Mag Sobel', fontsize=30)
            axs[2].imshow(dir_sobel, cmap='gray')
            axs[2].set_title('Dir Sobel', fontsize=30)
            axs[3].imshow(hls_s, cmap='gray')
            axs[3].set_title('HLS S', fontsize=30)
            axs[4].imshow(sobel_combine_, cmap='gray')
            axs[4].set_title('combine', fontsize=30)
            axs[5].imshow(combined, cmap='gray')
            axs[5].set_title('mag + dir', fontsize=30)
            plt.show()


        # Perspective Transform
        #img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

        # HLS L-channel Threshold (using default parameters)
        img_LThresh = hls_l_select(img_unwarp)

        # Lab B-channel Threshold (using default parameters)
        img_BThresh = lab_b_select(img_unwarp)

        # Combine HLS and Lab B channel thresholds
        combined = np.zeros_like(img_BThresh)
        combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
        return combined, Minv


        # image color and gradient thresholding
        #image_binary = combined_sobel_threshold(undistorted, mask_vertices=mask_vertices, c_thresh=c_thresh, gray_thresh=gray_thresh)
        #plt.imshow(image_binary, cmap='gray') if debug else None

image = cv2.cvtColor(cv2.imread("examples/signs_vehicles_xygrad.png"), cv2.COLOR_BGR2RGB)   # image was read as BGR, convert to rgb
Pipeline().run(image, debug=True)



"""
Pipeline:

1) camera calibration
2) undistort image
3) color and gradient threshold
4) perspective transform

5) find lines (with histogram)
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    plt.plot(histogram)

"""
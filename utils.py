import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle as pk
import os.path
import glob
from scipy.signal import find_peaks_cwt, general_gaussian, fftconvolve
from scipy.ndimage.measurements import center_of_mass
from collections import deque

# meters per pixel in y dimension
ym_per_pix = 30 / 720
# meters per pixel in x dimension
xm_per_pix = 3.7 / 700

time_window = 10        # results are averaged over this number of frames

class camera_calibration(object):

    def _get_calibration_params(self, calibration_images, debug=False):
        """Get calibration params using images in folder

        :param image_dir:
        :return:
        """
        nx = 9  # number of inside corners in x
        ny = 6  # number of inside corners in y

        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D ponts in image plane

        objp = np.zeros((ny * nx, 3), np.float32)  # second dimention is 3, for x, y, z coordinates
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # 3rd coordinate, Z, stays 0, as chessboard in on a place

        # cal_images_folder = image_dir
        # imagefiles = [f for f in listdir(cal_images_folder) if isfile(join(cal_images_folder, f))]
        # imagefiles = glob.glob(image_dir + sep + "calibration*.jpg")
        # Make a list of calibration images
        # calibration images folder
        if debug:
            fig, axs = plt.subplots(5, 4, figsize=(16, 11))
            fig.subplots_adjust(hspace=.2, wspace=.001)
            axs = axs.ravel()

        for i, img in enumerate(calibration_images):

            # img = cv2.imread(image)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # print(ret, corners)
            # If found, draw corners
            if ret == True:
                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                # plt.imshow(img)
                objpoints.append(objp)
                imgpoints.append(corners)

                if debug:
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    axs[i].axis('off')
                    axs[i].imshow(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return mtx, dist


    def calibration_params(self):
        if not os.path.exists("calibration.p"):
            # Load test and calibration images
            calibration_images = [cv2.imread(i) for i in glob.glob('./camera_Cal/*.jpg')]
            # get distortion parameters
            mtx, dist = self._get_calibration_params(calibration_images, True)
            calibration = {"mtx": mtx, "dist": dist}
            with open("calibration.p", "wb") as fh:
                pk.dump(calibration, fh)
        else:
            with open("calibration.p", "rb") as fh:
                calibration = pk.load(fh)

        return calibration['mtx'], calibration['dist']

class Line(object):
    def __init__(self, buffer_len=10):
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
        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = deque(maxlen=buffer_len)
        self.recent_fits_meter = deque(maxlen=2 * buffer_len)
        self.recent_fits = deque(maxlen=buffer_len)
        self.last_fit = None

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.
        :param new_fit_pixel: new polynomial coefficients (pixel)
        :param new_fit_meter: new polynomial coefficients (meter)
        :param detected: if the Line was detected or inferred
        :param clear_buffer: if True, reset state
        :return: None
        """
        self.detected = detected

        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    def update_fit(self, new_fit, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.
        :param new_fit_pixel: new polynomial coefficients (pixel)
        :param new_fit_meter: new polynomial coefficients (meter)
        :param detected: if the Line was detected or inferred
        :param clear_buffer: if True, reset state
        :return: None
        """
        self.detected = detected

        if clear_buffer:
            self.recent_fits = []

        self.last_fit = new_fit
        self.recent_fits.append(self.last_fit)

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # average of polynomial coefficients of the last N iterations
    def average_fits(self):
        return np.median(self.recent_fits, axis=0)

    @property
    # radius of curvature of the line (averaged)
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fits

        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


def get_fits_by_sliding_windows(unwarped_binarized_image, line_lt, line_rt, n_windows=9, debug=False):
    height, width = unwarped_binarized_image.shape

    histogram = np.sum(unwarped_binarized_image[height // 2:-30, :], axis=0)
    if debug:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((unwarped_binarized_image, unwarped_binarized_image, unwarped_binarized_image)) * 255
    else:
        out_img = None

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(height / n_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = unwarped_binarized_image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if debug:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    line_lt.all_x = leftx =  nonzero_x[left_lane_inds]
    line_lt.all_y = lefty = nonzero_y[left_lane_inds]
    line_rt.all_x = rightx = nonzero_x[right_lane_inds]
    line_rt.all_y = righty = nonzero_y[right_lane_inds]


    """
    Fit a second order polynomial to each, over the last n images

    """
    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)



    """
    Visualization
    """
    # Generate x and y values for plotting
    if debug:
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
        f, ax = plt.subplots(1, 2)
        f.set_facecolor('white')
        ax[0].imshow(unwarped_binarized_image, cmap='gray')
        ax[1].imshow(out_img)
        ax[1].plot(left_fitx, ploty, color='yellow')
        ax[1].plot(right_fitx, ploty, color='yellow')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)

        plt.show()

    return line_lt, line_rt, out_img



def undistort(image, mtx, dist):
    """
    undistort image using camera caliberation parameters
    """
    return cv2.undistort(image, mtx, dist, None, mtx)


def test_calibration(image, mtx, dist):
    # View undistorted calibration image
    img = cv2.imread("camera_cal/calibration4.jpg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    ax1.imshow(img)
    ax1.set_title('original image', fontsize=20)
    ax2.imshow(undistort(img, mtx, dist))
    ax2.set_title('undistorted mage', fontsize=20)
    plt.show()


def combined_sobel_threshold(image, mask_vertices, c_thresh, gray_thresh):
    """preprocessing: apply mask, binarization."""

    # mash image
    mask = np.uint8(np.zeros_like(image[:, :, 0]))
    vertices = mask_vertices
    cv2.fillPoly(mask, vertices, (1))

    # binarize in C channel, mainly detect yellow lane lines
    c_channel = np.max(image, axis=2) - np.min(image, axis=2)
    _, c_binary = cv2.threshold(c_channel, c_thresh[0], c_thresh[1], cv2.THRESH_BINARY)

    # binarize in gray channel, mainly detect white lane lines
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, gray_binary = cv2.threshold(gray, gray_thresh[0], gray_thresh[1], cv2.THRESH_BINARY)

    # combine the results
    combined_binary_masked = cv2.bitwise_and(cv2.bitwise_or(c_binary, gray_binary), mask)

    return combined_binary_masked

def visualize_color_channels(image):
    image_R = image[:,:,0]
    image_G = image[:,:,1]
    image_B = image[:,:,2]
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_H = image_HSV[:,:,0]
    image_S = image_HSV[:,:,1]
    image_V = image_HSV[:,:,2]
    image_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    image_L = image_LAB[:,:,0]
    image_A = image_LAB[:,:,1]
    image_B2 = image_LAB[:,:,2]
    fig, axs = plt.subplots(3,3, figsize=(16, 12))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    axs[0].imshow(image_R, cmap='gray')
    axs[0].set_title('RGB R', fontsize=30)
    axs[1].imshow(image_G, cmap='gray')
    axs[1].set_title('RGB G', fontsize=30)
    axs[2].imshow(image_B, cmap='gray')
    axs[2].set_title('RGB B', fontsize=30)
    axs[3].imshow(image_H, cmap='gray')
    axs[3].set_title('HSV H', fontsize=30)
    axs[4].imshow(image_S, cmap='gray')
    axs[4].set_title('HSV S', fontsize=30)
    axs[5].imshow(image_V, cmap='gray')
    axs[5].set_title('HSV V', fontsize=30)
    axs[6].imshow(image_L, cmap='gray')
    axs[6].set_title('LAB L', fontsize=30)
    axs[7].imshow(image_A, cmap='gray')
    axs[7].set_title('LAB A', fontsize=30)
    axs[8].imshow(image_B2, cmap='gray')
    axs[8].set_title('LAB B', fontsize=30)
    plt.show()

def abs_sobel_thresh(im, orient='x', thresh=(50, 200), sobel_kernel=3):
    img = np.copy(im)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # cv2.COLOR_RGB2GRAY if img read with mpimg.imread(), COLOR_BGR2GRAY for cv2

    # Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction)
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize=sobel_kernel)

    # Calculate the absolute value of the x derivative
    abs_sobel = np.absolute(sobel)

    # Convert the absolute value image to 8-bit
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a binary threshold to select pixels based on gradient strength
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0.01, np.pi/2), gray=False):
    # Grayscale
    if not gray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_s_select(img, thresh=(125, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def hls_l_select(img, thresh=(220, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def lab_b_select(img, thresh=(190,255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def sobel_combine(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

def threshold_single_channel(image, channel="R", thresh=(200, 255)):
    channels = {"R": image[:, :, 0],
    "G": image[:, :, 1],
    "B": image[:, :, 2]}
    single_channel = channels[channel]
    binary = np.zeros_like()
    binary[(single_channel > thresh[0]) & (single_channel <= thresh[1])] = 1
    return binary

def threshold_gray(image, thresh=(200, 255)):
    """
    detect white
    :param image:
    :param thresh:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, gray_binary = cv2.threshold(gray, thresh[0], thresh[1], cv2.THRESH_BINARY)
    return gray_binary

def C_channel_threshold(image, thresh=(89, 255)):
    """
    Detect yellow
    :param image:
    :param thresh:
    :return:
    """
    c_channel = np.max(image, axis=2) - np.min(image, axis=2)
    ret, c_binary = cv2.threshold(c_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    return c_binary


def preprocess(image, mask_vertices):
    """preprocessing: apply mask, binarization."""


    mask = np.uint8(np.zeros_like(image[:, :, 0]))
    cv2.fillPoly(mask, mask_vertices, (1))

    """
    # combine S channel of HLS and b channel of LAB
    # however, this does not give as good result as gray and C channel
    hls = hls_s_select(image)
    b =  lab_b_select(image)
    combined = np.zeros_like(b)
    combined[(hls == 1) | (b == 1)] = 1
    """

    gray_binary = threshold_gray(image)
    c_binary = C_channel_threshold(image)
    combined_binary_masked = (c_binary | gray_binary) & mask  # not that cv2 has bitwise_and and bitwise_or functions

    return combined_binary_masked

def get_unwarping_params(image):
    h, w = image.shape[:2]
    # define source and destination points for transform
    src = np.float32([(513, 483), (679, 475), (155, 685), (982, 685)])
    #dst = np.float32([(513, 483),  (513, 475), (155, 685), (155, 685)])
    dst = np.float32([(450, 0), (w - 450, 0), (450, h), (w - 450, h)])
    return (h, w), src, dst

def perspective_unwarp(img_binary,src,dst):
    """view from above perspective"""
    h, w = img_binary.shape[::-1]
    # transform Matric
    M = cv2.getPerspectiveTransform(src, dst)
    # inverse transform Matric
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img_binary, M, (h,w), flags=cv2.INTER_LINEAR)
    return warped, Minv


def select_lane_lines(binary_warped, n_windows=9, sliding_window=[80, 120], debug=False):
    """select pixels that are lane lines"""


    height, width = binary_warped.shape

    start = True
    if start:
        window_width = sliding_window[0]
        window_height = sliding_window[1]

        # histogram of the lower half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        if debug:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # find the peaks from the smoothed curve
        peakidx = find_peaks_cwt(histogram, np.arange(90, 100))

        # if more than two peaks found, report the error
        if len(peakidx) != 2:
            print('peakidx len is {}'.format(peakidx))


        # slidding window
        left_windows = []
        right_windows = []
        current_left_window = [peakidx[0] - window_width, peakidx[0] + window_width]
        current_right_window = [peakidx[1] - window_width, peakidx[1] + window_width]

        for window in range(n_windows):
            # update current sliding window according to the center of mass in the current window
            current_height_window = [binary_warped.shape[0] - (window + 1) * window_height, binary_warped.shape[0] - window * window_height]

            left_lane_windowed = binary_warped[current_height_window[0]:current_height_window[1],
                                 current_left_window[0]:current_left_window[1]]
            right_lane_windowed = binary_warped[current_height_window[0]:current_height_window[1],
                                  current_right_window[0]:current_right_window[1]]

            #cv2.rectangle(out_img, (current_left_window - 100, current_height_window[0]), (current_left_window+100, current_height_window[1]), (0, 255, 0), 2)
            #cv2.rectangle(out_img, (current_right_window-100, current_height_window[0]), (current_right_window+100, current_height_window[1]), (0, 255, 0), 2)

            hist_left = np.sum(left_lane_windowed, axis=0)
            hist_right = np.sum(right_lane_windowed, axis=0)
            # if there is no pixels in the current window, skip the updating
            if not all(hist_left == 0):
                center_left = int(center_of_mass(hist_left)[0]) + current_left_window[0]
                current_left_window = [center_left - window_width, center_left + window_width]
            if not all(hist_right == 0):
                center_right = int(center_of_mass(hist_right)[0]) + current_right_window[0]
                current_right_window = [center_right - window_width, center_right + window_width]
            # record the current window
            left_windows.append([current_height_window, current_left_window])
            right_windows.append([current_height_window, current_right_window])

        left_window_binary = np.zeros_like(binary_warped)
        right_window_binary = np.zeros_like(binary_warped)
        for wd in left_windows:
            xmin = wd[0][0]
            xmax = wd[0][1]
            ymin = wd[1][0]
            ymax = wd[1][1]
            left_window_binary[xmin:xmax, ymin:ymax] = 1

        for wd in right_windows:
            xmin = wd[0][0]
            xmax = wd[0][1]
            ymin = wd[1][0]
            ymax = wd[1][1]
            right_window_binary[xmin:xmax, ymin:ymax] = 1

        # select only the pixels in the sliding window
        left_lane_binary = np.zeros_like(binary_warped)
        left_lane_binary[(left_window_binary == 1) & (binary_warped == 1)] = 1
        right_lane_binary = np.zeros_like(binary_warped)
        right_lane_binary[(right_window_binary == 1) & (binary_warped == 1)] = 1


    else:
        height, width = binary_warped.shape

        left_fit_pixel = ll.last_fit_pixel
        right_fit_pixel = rl.last_fit_pixel

        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        margin = 100
        left_lane_inds = (
        (nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & (
        nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
        right_lane_inds = (
        (nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & (
        nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))


    return [left_lane_binary, right_lane_binary]


def fit_lane_line(lr_binary_images):
    """fit the left and right lane lines, return the quadratic coefficients"""
    left_lane_binary = lr_binary_images[0]
    right_lane_binary = lr_binary_images[1]

    # fit the left and right lane pixels
    left_Y, left_X = np.where(left_lane_binary == 1)
    left_fit = np.polyfit(left_Y, left_X, 2)
    #left_fitx = left_fit[0] * left_Y ** 2 + left_fit[1] * left_Y + left_fit[2]

    right_Y, right_X = np.where(right_lane_binary == 1)
    right_fit = np.polyfit(right_Y, right_X, 2)
    #right_fitx = right_fit[0] * right_Y ** 2 + right_fit[1] * right_Y + right_fit[2]

    return [left_fit, right_fit]


def cal_curvature(lr_binary_images):
    y_eval = 720

    left_lane_binary = lr_binary_images[0]
    right_lane_binary = lr_binary_images[1]

    lefty, leftx = np.where(left_lane_binary == 1)
    righty, rightx = np.where(right_lane_binary == 1)

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    # average radius between left and right lanes
    radius = (left_curverad + right_curverad) / 2.

    # fit polynomials in pixel space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # left right lane position
    left_lane_pos = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_lane_pos = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

    # diviate from center in meters
    deviation = ((right_lane_pos + left_lane_pos) / 2. - left_lane_binary.shape[1] / 2.) * xm_per_pix

    return [radius, deviation]


def draw_lane_lines(mask, color=(255, 0, 0), line_width=20, coeffs=None):
    """
    Draw the line on a color mask image.
    """
    h, w, c = mask.shape

    plot_y = np.linspace(0, h - 1, h)
    #coeffs = self.average_fit if average else self.last_fit_pixel

    line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
    line_left_side = line_center - line_width // 2
    line_right_side = line_center + line_width // 2

    # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array(list(zip(line_left_side, plot_y)))
    pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
    pts = np.vstack([pts_left, pts_right])

    # Draw the lane onto the warped blank image
    return cv2.fillPoly(mask, [np.int32(pts)], color)


def draw_lane(img_undistorted, img_binary, unwarp, Minv, left_right_fits, ll, rl,):
    """highlight lane using left and right fitting parameters"""


    height, width, _ = img_undistorted.shape
    h, w = height, width

    left_fit = left_right_fits[0]
    right_fit = left_right_fits[1]

    # y values
    yval = np.linspace(0, img_undistorted.shape[0])

    ll.update_fit(left_fit, True)
    rl.update_fit(right_fit, True)
    left_fit, right_fit = ll.average_fits, rl.average_fits

    # left and right x values
    left_fitx = left_fit[0] * yval ** 2 + left_fit[1] * yval + left_fit[2]
    right_fitx = right_fit[0] * yval ** 2 + right_fit[1] * yval + right_fit[2]

    # left, right lane lines according to the fitting
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yval]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yval])))])
    pts = np.hstack((pts_left, pts_right))

    # draw polygon
    road_warp = np.zeros_like(img_undistorted).astype(np.uint8)


    # Draw the windows on the visualization image
    #cv2.rectangle(color_warp, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    #cv2.rectangle(color_warp, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (img_undistorted.shape[1], img_undistorted.shape[0]))
    result = cv2.addWeighted(img_undistorted, 1, road_dewarped, 0.3, 0)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = draw_lane_lines(line_warp, color=(255, 0, 0), coeffs = left_fit)
    line_warp = draw_lane_lines(line_warp, color=(0, 0, 255), coeffs = right_fit)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))


    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15
    mask = np.zeros_like(line_dewarped).astype(np.uint8)
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(100, 100, 100), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=line_dewarped, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h + off_y, off_x:off_x + thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(unwarp, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h + off_y, 2 * off_x + thumb_w:2 * (off_x + thumb_w), :] = thumb_birdeye


    lines_mask = np.zeros_like(line_dewarped).astype(np.uint8)
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]
    blend_on_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_on_road, beta=0.5, gamma=0.)

    #lines_mask = np.zeros_like(line_dewarped).astype(np.uint8)
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]
    result = cv2.addWeighted(src1=blend_on_road, alpha=0.8, src2=result, beta=0.5, gamma=0.)


    return result

def write_radius_offset(image, radius, deviation):
    img = np.copy(image)
    cv2.putText(img, "curve radius: {} m".format(int(radius)), (image.shape[1]-400,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,225), 2, cv2.LINE_AA)
    cv2.putText(img, "Center Offset: {0:.2f} m".format(deviation), (image.shape[1]-400,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,225), 2, cv2.LINE_AA)
    return img


def lane_detection(img,mtx,dist,mask_vertices, src, dst,sliding_window, ll, rl, debug=False):
    """combined workflow of detecting lane"""

    image = np.copy(img)
    undist_image = undistort(image,mtx,dist)

    image_binary = preprocess(undist_image, mask_vertices )
    unwarp, MinV = perspective_unwarp(image_binary, src, dst)


    (il,ir)=select_lane_lines(unwarp, 9, sliding_window, False)

    if debug:
        fig, axs = plt.subplots(2, 3, figsize=(16, 4))
        axs = axs.ravel()
        axs[0].imshow(il, cmap='gray')
        axs[0].set_title('left line')
        axs[1].imshow(ir, cmap='gray')
        axs[1].set_title('right line')

        axs[2].imshow(image_binary, cmap='gray')
        axs[2].set_title('lanes')

        axs[3].imshow(unwarp, cmap='gray')
        axs[3].set_title('lanes')

    image_highlight = draw_lane(undist_image, image_binary, unwarp,  MinV, fit_lane_line((il,ir)), ll, rl,)
    radius = (ll.curvature + rl.curvature) / 2
    _, deviation = cal_curvature([il,ir])
    result = write_radius_offset(image_highlight, radius, deviation)

    if debug:
        axs[4].imshow(result, cmap='gray')
        axs[4].set_title('result')
        plt.show()

    return result



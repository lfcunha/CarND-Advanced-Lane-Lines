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

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

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
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(img)
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
        self.last_fit = None
        #self.curvature_deviation = None

    def update_fit(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
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
        self.recent_fits_pixel.append(new_fit_pixel)
        self.recent_fits_meter.append(new_fit_meter)

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.median_fit if average else self.last_fit_pixel

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
    # median of polynomial coefficients of the last N iterations
    def median_fit(self):
        return {"median_fit_pixel": np.median(self.recent_fits_pixel, axis=0), "median_pixel_meter": np.median(self.recent_fits_meter, axis=0)}

    @property
    # radius of curvature of the line, and offset from center
    def curvature_deviation(self):
        #TODO: Can't calculate directly on the stored fits. Needs to reverse the pixels first
        raise NotImplementedError()
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        y_eval = 720
        coeffs = self.median_fit.get("median_pixel_meter")
        coeffs_pixel = self.median_fit.get("median_fit_pixel")
        # Calculate the new radii of curvature

        # left right lane position
        lane_pos = coeffs_pixel[0] * y_eval ** 2 + coeffs_pixel[1] * y_eval + coeffs_pixel[2]

        return ((1 + (2 * coeffs[0] * y_eval * ym_per_pix +
                               coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0]), lane_pos


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


def B_channel_threshold(image, thresh=(160, 255)):
    """
    Detect yellow
    :param image:
    :param thresh:
    :return:
    """
    # c_channel = np.max(image, axis=2) - np.min(image, axis=2)
    # b nd c channels give similar results.

    #optimal thresholds to use with each channel type
    #b channel: (160, 255)
    #c challen: (90, 255)

    b_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 2]
    ret, c_binary = cv2.threshold(b_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    return c_binary


def pre_process_image(image, mask_vertices):
    """
    preprocessing: apply mask, binarization.
    """

    mask = np.uint8(np.zeros_like(image[:, :, 0]))
    cv2.fillPoly(mask, mask_vertices, (1))

    """
    # combine S channel of HLS and b channel of LAB
    # however, this does not give as good result as gray and C/b channels
    hls = hls_s_select(image)
    b =  lab_b_select(image)
    combined = np.zeros_like(b)
    combined[(hls == 1) | (b == 1)] = 1
    """

    gray_binary = threshold_gray(image)
    b_binary = B_channel_threshold(image)

    combined_binary_masked = (b_binary | gray_binary) & mask  # not that cv2 has bitwise_and and bitwise_or functions

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
    unwarped = cv2.warpPerspective(img_binary, M, (h,w), flags=cv2.INTER_LINEAR)
    return unwarped, Minv

def select_lane_lines(binary_warped, n_windows=9, sliding_window=[80, 120], debug=False):
    """select pixels that are lane lines"""

    start = True

    if start:
        window_width = sliding_window[0]
        window_height = sliding_window[1]

        # histogram of the lower half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        if debug:
            pass
            # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # find the peaks from the smoothed curve
        peakindices = find_peaks_cwt(histogram, np.arange(90, 100))

        # slidding window
        left_search_windows = []
        right_search_windows = []
        current_left_window = [peakindices[0] - window_width, peakindices[0] + window_width]
        current_right_window = [peakindices[1] - window_width, peakindices[1] + window_width]

        for window in range(n_windows):
            # update current sliding window according to the center of mass in the current window
            current_height_window = [binary_warped.shape[0] - (window + 1) * window_height, binary_warped.shape[0] - window * window_height]

            left_lane_windowed = binary_warped[current_height_window[0]:current_height_window[1],
                                 current_left_window[0]:current_left_window[1]]
            right_lane_windowed = binary_warped[current_height_window[0]:current_height_window[1],
                                  current_right_window[0]:current_right_window[1]]

            if debug:
                # draw rectangles
                cv2.rectangle(out_img, (current_left_window[0] - 100, current_height_window[0]), (current_left_window[1]+100, current_height_window[1]), (0, 255, 0), 2)
                cv2.rectangle(out_img, (current_right_window[0]-100, current_height_window[0]), (current_right_window[1]+100, current_height_window[1]), (0, 255, 0), 2)

            histogram_left, histogram_right = np.sum(left_lane_windowed, axis=0), np.sum(right_lane_windowed, axis=0)
            # if there is no pixels in the current window, skip the updating
            if all(histogram_left == 0):
                pass
            else:
                center_left = int(center_of_mass(histogram_left)[0]) + current_left_window[0]
                current_left_window = [center_left - window_width, center_left + window_width]
            if all(histogram_right == 0):
                pass
            else:
                center_right = int(center_of_mass(histogram_right)[0]) + current_right_window[0]
                current_right_window = [center_right - window_width, center_right + window_width]
            # record the current window
            left_search_windows.append([current_height_window, current_left_window])
            right_search_windows.append([current_height_window, current_right_window])

        if debug:
            #plt.imshow(out_img)
            #plt.show()
            pass

        left_window_binary = np.zeros_like(binary_warped)
        right_window_binary = np.zeros_like(binary_warped)
        for window in left_search_windows:
            x_min, x_max, y_min, y_max = window[0][0], window[0][1], window[1][0], window[1][1]
            left_window_binary[x_min:x_max, y_min:y_max] = 1

        for window in right_search_windows:
            x_min, x_max, y_min, y_max = window[0][0], window[0][1], window[1][0], window[1][1]
            right_window_binary[x_min:x_max, y_min:y_max] = 1

        # select only the pixels in the sliding window
        ll_binary = np.zeros_like(binary_warped)
        ll_binary[(left_window_binary == 1) & (binary_warped == 1)] = 1
        rl_binary = np.zeros_like(binary_warped)
        rl_binary[(right_window_binary == 1) & (binary_warped == 1)] = 1

    else:
        # TODO: implement boxing around line fit (+- margin) for "inteligent" detection guided by previous blind detection
        pass

    return [ll_binary, rl_binary]


def fit_lane(binary_images, ll, rl):
    """
    fit lines to both lanes

    :param binary_images:
    :param ll: left Line instance
    :param rl: right Line instance
    :return: quadratic coefficients
    """
    left_lane = binary_images[0]
    right_lane = binary_images[1]

    # select and fit the left lane pixels
    left_Y, left_X = np.where(left_lane == 1)
    left_fit_pixel = np.polyfit(left_Y, left_X, 2)
    #left_fitx = left_fit[0] * left_Y ** 2 + left_fit[1] * left_Y + left_fit[2]

    # select and fit the right lane pixels
    right_Y, right_X = np.where(right_lane == 1)
    right_fit_pixel = np.polyfit(right_Y, right_X, 2)
    #right_fitx = right_fit[0] * right_Y ** 2 + right_fit[1] * right_Y + right_fit[2]

    """
    ll.allx = left_X
    ll.ally = left_Y
    rl.allx = right_X
    rl.ally = right_Y

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    #leftx_inv = left_X[::-1]  # Reverse to match top-to-bottom in y
    #rightx_inv = right_X[::-1]  # Reverse to match top-to-bottom in y
    """

    # Fit new polynomials to x,y in world space
    left_fit_meter = np.polyfit(left_Y * ym_per_pix, left_Y * xm_per_pix, 2)
    right_fit_meter = np.polyfit(right_Y * ym_per_pix, right_Y * xm_per_pix, 2)
    #right_fit_meter_inv = np.polyfit(right_Y * ym_per_pix, right_Y * xm_per_pix, 2)
    return [(left_fit_pixel, right_fit_pixel),(left_fit_meter, right_fit_meter)]


def calculate_curvature_and_deviation(binary_images):
    """
    calculare average curvature and deviation from center of the two lanes
    :param binary_images:
    :return: float, float radius, deviation
    """

    # Y value at the bottom of the image
    y_eval = binary_images[0].shape[0]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_lane = binary_images[0]
    right_lane = binary_images[1]

    lefty, leftx = np.where(left_lane == 1)
    righty, rightx = np.where(right_lane == 1)

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # calculate the average radius of the two lanes
    radius = (left_curverad + right_curverad) / 2.

    # fit polynomials in pixel space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # deviation from center in meters
    _Centeroffset = ((right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2] + left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]) / 2. - left_lane.shape[1] / 2.) * xm_per_pix

    return radius, _Centeroffset


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


def draw_lane(img_undistorted, img_binary, unwarp, Minv, left_right_fits, ll, rl, il, ir, debug=False):
    """highlight lane using left and right fitting parameters"""


    height, width, _ = img_undistorted.shape
    h, w = height, width

    left_fit_pixel = left_right_fits[0][0]
    right_fit_pixel = left_right_fits[0][1]
    left_fit_meter = left_right_fits[1][0]
    right_fit_meter = left_right_fits[1][1]

    # y values
    ploty = np.linspace(0, img_undistorted.shape[0])

    ll.update_fit(left_fit_pixel, left_fit_meter, True)
    rl.update_fit(right_fit_pixel, right_fit_meter,True)

    left_fit_pixel, right_fit_pixel = ll.median_fit.get("median_fit_pixel"), rl.median_fit.get("median_fit_pixel")

    # left and right x values
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # left, right lane lines according to the fitting
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    out_img = np.dstack((unwarp, unwarp, unwarp)) * 255
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    if debug:
        nonzero = unwarp.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        out_img[nonzero_y[il], nonzero_x[il]] = [255, 0, 0]
        out_img[nonzero_y[ir], nonzero_x[ir]] = [0, 0, 255]
        f, ax = plt.subplots(1, 2)
        f.set_facecolor('white')
        ax[0].imshow(unwarp, cmap='gray')
        ax[0].set_title('detected line pixels')
        ax[1].imshow(out_img)
        ax[1].plot(left_fitx, ploty, color='red')
        ax[1].plot(right_fitx, ploty, color='blue')
        ax[1].set_title('right/left line fit')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)
        plt.show()

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
    line_warp = draw_lane_lines(line_warp, color=(255, 0, 0), coeffs = left_fit_pixel)
    line_warp = draw_lane_lines(line_warp, color=(0, 0, 255), coeffs = right_fit_pixel)

    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    if debug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        ax1.imshow(line_dewarped, cmap='gray')
        ax1.set_title('left line')
        plt.show()

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15
    mask = np.zeros_like(line_dewarped).astype(np.uint8)
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(100, 100, 100), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=result, beta=0.8, gamma=0)

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

    result = cv2.addWeighted(src1=blend_on_road, alpha=0.8, src2=result, beta=0.5, gamma=0.)

    """
    # TODO: implement reverse pixels in curvature_deviation()
    curve_r, rp = rl.curvature_deviation
    curve_l, lp = ll.curvature_deviation
    curvature = (curve_r + curve_l)/2
    deviation = ((rp + lp) / 2. - img_binary.shape[1] / 2.) * xm_per_pix
    print(curvature, deviation)
    """

    return result


def detect_lane(img, mtx, dist, mask_vertices, src, dst, sliding_window, ll, rl, debug=False):
    """combined workflow of detecting lane"""

    image = np.copy(img)
    undist_image = undistort(image,mtx,dist)

    image_binary = pre_process_image(undist_image, mask_vertices)
    unwarp, MinV = perspective_unwarp(image_binary, src, dst)

    (il,ir) = select_lane_lines(unwarp, 9, sliding_window, False)

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

    image_annotated_lanes = draw_lane(undist_image, image_binary, unwarp, MinV, fit_lane((il, ir), ll, rl), ll, rl, il, ir)
    radius, deviation = calculate_curvature_and_deviation([il, ir])

    img = np.copy(image_annotated_lanes)
    cv2.putText(img, "curve radius: {} m".format(int(radius)), (image.shape[1]-400,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,225), 2, cv2.LINE_AA)
    cv2.putText(img, "Center Offset: {0:.2f} m".format(deviation), (image.shape[1]-400,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,225), 2, cv2.LINE_AA)


    if debug:
        axs[4].imshow(img, cmap='gray')
        axs[4].set_title('result')
        plt.show()

    return img



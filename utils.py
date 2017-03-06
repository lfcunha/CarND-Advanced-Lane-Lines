import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle as pk
import os.path
import glob

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
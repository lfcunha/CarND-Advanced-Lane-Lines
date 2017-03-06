import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import listdir, sep
from os.path import isfile, join
import glob

# prepare object points
nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y



def get_calibration_params(calibration_images):
    """Get calibration params using images in folder

    :param image_dir:
    :return:
    """
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D ponts in image plane

    objp = np.zeros((ny*nx, 3), np.float32)  # second dimention is 3, for x, y, z coordinates
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)   # 3rd coordinate, Z, stays 0, as chessboard in on a place

    #cal_images_folder = image_dir
    #imagefiles = [f for f in listdir(cal_images_folder) if isfile(join(cal_images_folder, f))]
    #imagefiles = glob.glob(image_dir + sep + "calibration*.jpg")
    # Make a list of calibration images
    # calibration images folder

    for img in calibration_images:

        #img = cv2.imread(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        #print(ret, corners)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return mtx, dist


def undistort(image, mtx, dist):
    """
    undistort image using camera caliberation parameters
    """
    return cv2.undistort(image, mtx,dist, None, mtx)


"""
#############################
"""

"""
# Load images
test_images = [cv2.imread(i) for i in glob.glob('./test_images/*.jpg')]
calibration_images = [cv2.imread(i) for i in glob.glob('./camera_Cal/*.jpg')]

# get distortion parameters
mtx, dist = get_calibration_params(calibration_images)

# undistort image

# View undistorted calibration image
img = cv2.imread("camera_cal/calibration4.jpg")
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,4))
ax1.imshow(img)
ax1.set_title('original image',fontsize=20)
ax2.imshow(undistort(img,mtx,dist))
ax2.set_title('undistorted mage',fontsize=20)
plt.show()

# View undistorted test image
img = test_images[0]
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,4))
ax1.imshow(img, cmap="gray")
ax1.set_title('original image',fontsize=20)
ax2.imshow(undistort(img,mtx,dist), cmap="gray")
ax2.set_title('undistorted mage',fontsize=20)
plt.show()
"""

# Threshold image
def abs_sobel_thresh(im, orient='x', thresh=(0, 255), sobel_kernel=3):
    img = np.copy(im)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # cv2.COLOR_RGB2GRAY if img read with mpimg.imread(), COLOR_BGR2GRAY for cv2

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

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def color_threshold(image):
    thresh = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary

def threshold_single_channel(image, channel="R"):
    channels = {"R": image[:, :, 0],
    "G": image[:, :, 1],
    "B": image[:, :, 2]}
    single_channel = channels[channel]
    thresh = (200, 255)
    binary = np.zeros_like()
    binary[(single_channel > thresh[0]) & (single_channel <= thresh[1])] = 1
    return binary


def hls_threshold(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    thresh = (90, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# pipeline to combine color and gradient threshold.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
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



# plt.imshow(sxbinary, cmap='gray')

# View undistorted calibration image
image = cv2.imread("examples/signs_vehicles_xygrad.png")

result = pipeline(image)

# Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()

#ax1.imshow(image)
#ax1.set_title('Original Image', fontsize=40)


hls_binary = hls_select(image, thresh=(90, 255))

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,4))
ax1.imshow(image)
ax1.set_title('original image',fontsize=20)

ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0.1, 0.7))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0.1, 0.7))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0.2, 1.7))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

ax2.imshow(result, cmap='gray')
ax2.set_title('thresholded',fontsize=20)
plt.show()


# Perspective transform

src, dst = None, None

def wrap(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[0,0], [0,1], [-1,0], [-1,-1]])
    dst = np.float32([[0, 0], [0, 1], [-1, 0], [-1, -1]])

    # Compute the perspective transform, M, given source and destination points
    M = cv2.getPerspectiveTransform(src, dst)

    # Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp an image using the perspective transform, M
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped



"""
1) camera calibration
2) undistort image
3) color and gradient threshold
4) perspective transform

5) find lines (with histogram)
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    plt.plot(histogram)

"""
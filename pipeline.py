from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip


class Pipeline(object):
    def __init__(self):
        pass

    def process_image(self, image=None, debug=False):
        """Process an image, creating visual outputs to understand the process, including visualization of
        color channels and thresholding. For video processing, use the process_video function

        :param image:
        :param debug:
        :return:
        """

        # get camera calibration parameters
        mtx,dist = camera_calibration().calibration_params()

        if debug:
            # inspect calibration
            test_images = [cv2.imread(i) for i in glob.glob('./test_images/*.jpg')]
            test_calibration(test_images[0], mtx,dist)

        # Undistort image
        undistorted = undistort(image, mtx=mtx, dist=dist)

        imshape = image.shape
        # create mask on center of image to identify lanes
        # the relative proportion of the image to decide the vertives was determined empirically
        mask_vertices = np.array([[(imshape[1] / 8, imshape[0]), (2 * imshape[1] / 6, 2 * (imshape[0] / 5)),
                                   (7 * imshape[1] / 12, 2 * (imshape[0] / 5)), ((11 * imshape[1] / 12), imshape[0])
                                   ]], dtype=np.int32)

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

            preprocessed = pre_process_image(undistorted, mask_vertices)

            fig, axs = plt.subplots(4, 3, figsize=(16, 12))
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
            axs[6].imshow(preprocessed, cmap='gray')
            axs[6].set_title('preprocessed', fontsize=30)

        # mask image to select only center + do color selection
        binary_image = pre_process_image(undistorted, mask_vertices)

        size, src, dst = get_unwarping_params(image)
        h, w = size

        # change perspective
        unwarp, MinV = perspective_unwarp(binary_image, src, dst)

        if debug:
            # Visualize bird eye view perpective (unwarped image)
            axs[7].imshow(undistorted)
            x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
            y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
            axs[7].plot(x, y, color='#32cc98', alpha=0.4, linewidth=2, solid_capstyle='round', zorder=2)
            axs[7].set_ylim([h, 0])
            axs[7].set_xlim([0, w])
            axs[7].set_title('Undistorted Image', fontsize=30)
            axs[8].imshow(unwarp, cmap='gray')
            axs[8].set_title('Unwarped Image', fontsize=30)

        #line_lt, line_rt = Line(buffer_len=10), Line(buffer_len=10)
        #left_lane, right_lane, img_out = get_fits_by_sliding_windows(unwarp, line_lt, line_rt, 7, True)

        left_lane, right_lane = select_lane_lines(unwarp)

        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
            axs[9].imshow(left_lane, cmap='gray')
            axs[9].set_title('left line')
            axs[10].imshow(right_lane, cmap='gray')
            axs[10].set_title('right line')

        if debug:
            plt.show()

        left_fit, right_fit = fit_lane([left_lane, right_lane])

        yval = np.linspace(0, binary_image.shape[1])
        left_Y, left_X = np.where(left_lane == 1)
        left_fitx = left_fit[0] * yval ** 2 + left_fit[1] * yval + left_fit[2]

        right_Y, right_X = np.where(right_lane == 1)
        right_fitx = right_fit[0] * yval ** 2 + right_fit[1] * yval + right_fit[2]

        plt.plot(left_X, left_Y, '+')
        plt.plot(right_X, right_Y, '+')
        plt.plot(left_fitx, yval)
        plt.plot(right_fitx, yval)
        plt.gca().invert_yaxis()

        radius, deviation = calculate_curvature_and_deviation([left_lane, right_lane])
        image_annotated_lanes = draw_lane(image, [left_fit, right_fit], src, dst)
        plt.imshow(image_annotated_lanes)

        img = np.copy(image_annotated_lanes)
        cv2.putText(img, "curve radius: {} m".format(int(radius)), (image.shape[1] - 400, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (225, 225, 225), 2, cv2.LINE_AA)
        cv2.putText(img, "Center Offset: {0:.2f} m".format(deviation), (image.shape[1] - 400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2, cv2.LINE_AA)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        ax1.imshow(image)
        ax2.imshow(img)

        return unwarp

    def detect_lane(self, img, mtx, dist, mask_vertices, src, dst, sliding_window, ll, rl, debug=False):
        """combined workflow of detecting lane"""

        image = np.copy(img)
        undist_image = undistort(image, mtx, dist)

        image_binary = pre_process_image(undist_image, mask_vertices)
        unwarp, MinV = perspective_unwarp(image_binary, src, dst)

        (il, ir) = select_lane_lines(unwarp, 9, sliding_window, False)

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

        image_annotated_lanes = draw_lane(undist_image, image_binary, unwarp, MinV, fit_lane((il, ir), ll, rl), ll, rl,
                                          il, ir)
        radius, deviation = calculate_curvature_and_deviation([il, ir])

        img = np.copy(image_annotated_lanes)
        cv2.putText(img, "curve radius: {} m".format(int(radius)), (image.shape[1] - 400, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (225, 225, 225), 2, cv2.LINE_AA)
        cv2.putText(img, "Center Offset: {0:.2f} m".format(deviation), (image.shape[1] - 400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2, cv2.LINE_AA)

        if debug:
            axs[4].imshow(img, cmap='gray')
            axs[4].set_title('result')
            plt.show()

        return img

    def process_frame(sefl, image):
        return detect_lane(image, mtx, dist, mask_vertices, src, dst, sliding_window, ll, rl, debug=False)

    def process_video(self, video_filename):
        filename_, ext = video_filename.split(".")
        project_video_output = filename_ + "_output" + "." + ext
        clip1 = VideoFileClip(video_filename)
        lane_clip = clip1.fl_image(self.process_frame)  # NOTE: this function expects color images!!
        lane_clip.write_videofile(project_video_output, audio=False)


image = cv2.cvtColor(cv2.imread("test_images/test1.jpg"), cv2.COLOR_BGR2RGB)   # image was read as BGR, convert to rgb

############
img = Pipeline().process_image(image, debug=True)  #####################
############

#plt.imshow(img, cmap="gray")
#plt.show()

imshape = image.shape
# create mask on center of image to identify lanes
# the relative proportion of the image to decide the vertives was determined empirically
mask_vertices = np.array([[(imshape[1] / 8, imshape[0]),
                           (2 * imshape[1] / 6, 2 * (imshape[0] / 5)),
                           (7 * imshape[1] / 12, 2 * (imshape[0] / 5)),
                           ((11 * imshape[1] / 12), imshape[0])
                           ]], dtype=np.int32)

mtx,dist = camera_calibration().calibration_params()

size, src, dst = get_unwarping_params(image)
#h, w = size

ll, rl = Line(), Line()

sliding_window=[80,120]

def frame_func(image, debug=False):
    return detect_lane(image, mtx, dist, mask_vertices, src, dst, sliding_window, ll, rl, debug)



#plt.show()

def process_video():
    project_video_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    lane_clip = clip1.fl_image(frame_func)  # NOTE: this function expects color images!!
    lane_clip.write_videofile(project_video_output, audio=False)

do = 0

if do:
    process_video()
else:
    frame_func(image, True)


def visualize_perpective_transformation_on_test_images():
    # Make a list of example images
    images = glob.glob('./test_images/*.jpg')

    # Set up plot
    fig, axs = plt.subplots(len(images), 2, figsize=(10, 20))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()

    i = 0
    for image in images:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bin = Pipeline().process_image(img)
        axs[i].imshow(img)
        axs[i].axis('off')
        i += 1
        axs[i].imshow(img_bin, cmap='gray')
        axs[i].axis('off')
        i += 1
    plt.show()

#visualize_perpective_transformation_on_test_images()

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
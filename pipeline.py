from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


"""
Setup:
- Define image mask region
- Get/save camera calibration parameters
"""

image = cv2.cvtColor(cv2.imread("test_images/test1.jpg"), cv2.COLOR_BGR2RGB)   # image was read as BGR, convert to rgb
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


class Pipeline(object):
    image = cv2.cvtColor(cv2.imread("test_images/test1.jpg"),
                         cv2.COLOR_BGR2RGB)  # image was read as BGR, convert to rgb
    imshape = image.shape
    # create mask on center of image to identify lanes
    # the relative proportion of the image to decide the vertives was determined empirically
    mask_vertices = np.array([[(imshape[1] / 8, imshape[0]),
                               (2 * imshape[1] / 6, 2 * (imshape[0] / 5)),
                               (7 * imshape[1] / 12, 2 * (imshape[0] / 5)),
                               ((11 * imshape[1] / 12), imshape[0])
                               ]], dtype=np.int32)
    mtx, dist = camera_calibration().calibration_params()
    size, src, dst = get_unwarping_params(image)
    def __init__(self):
        pass

    def detect_lane(self, img, mtx, dist, mask_vertices, src, dst, sliding_window=[80,120], ll=None, rl=None, debug=False):
        """Detect the left and right lanes in the image
        :param img:
        :param mtx:
        :param dist:
        :param mask_vertices:
        :param src:
        :param dst:
        :param sliding_window:
        :param ll:
        :param rl:
        :param debug:
        :return:
        """

        image = np.copy(img)

        # undistort the image using the camera calibration parameters
        undist_image = undistort(image, mtx, dist)

        # threshold, mask, and binarize the image
        image_binary = pre_process_image(undist_image, mask_vertices)

        # change the perspective of the image (to bird-eye view)
        unwarp, MinV = perspective_unwarp(image_binary, src, dst)

        # detect the lane pixels in the image. Return two images, one for each lane
        left_lane_pixels, right_lane_pixels = select_lane_lines(unwarp, 9, sliding_window, False)

        if debug:
            fig, axs = plt.subplots(2, 3, figsize=(16, 4))
            axs = axs.ravel()
            axs[0].imshow(left_lane_pixels, cmap='gray')
            axs[0].set_title('left line')
            axs[1].imshow(right_lane_pixels, cmap='gray')
            axs[1].set_title('right line')
            axs[2].imshow(image_binary, cmap='gray')
            axs[2].set_title('lanes')
            axs[3].imshow(unwarp, cmap='gray')
            axs[3].set_title('lanes')

        # Detect lane, fit a quadratic + draw polygon highligting the lane
        image_annotated_lanes = draw_lane(undist_image, image_binary, unwarp, MinV, fit_lane((left_lane_pixels, right_lane_pixels), ll, rl), ll, rl,
                                          left_lane_pixels, right_lane_pixels)

        # calculate the radius and deviation from the center of the road
        radius, deviation = calculate_curvature_and_deviation([left_lane_pixels, right_lane_pixels])

        # write the radius and deviation to the frame
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
        # create line instances for the left and right lanes, which will be used to track some information, such as lane fits over several frames
        ll, rl = Line(), Line()
        sliding_window = [80, 120]
        return detect_lane(image, mtx, dist, mask_vertices, src, dst, sliding_window, ll, rl, debug=False)

    def process_video(self, video_filename):
        filename_, ext = video_filename.split(".")
        project_video_output = filename_ + "_output" + "." + ext
        clip1 = VideoFileClip(video_filename)
        lane_clip = clip1.fl_image(self.process_frame)  # NOTE: this function expects color images!!
        lane_clip.write_videofile(project_video_output, audio=False)


Pipeline().process_video("project_video.mp4")

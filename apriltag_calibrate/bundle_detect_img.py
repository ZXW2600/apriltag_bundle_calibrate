import apriltag
import cv2
from matplotlib import pyplot as plt

import numpy as np
import argparse

from tqdm import tqdm

from apriltag_calibrate.utils.ImageLoader import ImageLoader
from apriltag_calibrate.configparase import TagBundle, Camera
from apriltag_calibrate.utils.TagPnp import TagPnP
from apriltag_calibrate.visualization import draw_axes, draw_camera
from apriltag_calibrate.utils.Geometery import Rtvec2HomogeousT


# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image folder")
ap.add_argument("-c", "--camera", required=True,
                help="path to the camera calibration file")
ap.add_argument("-b", "--bundle", required=True,
                help="path to the bundle calibration file")

args = ap.parse_args()
image_path = args.image
camera_param_path = args.camera
bundle_param_path = args.bundle

# read the image
imageset = ImageLoader(image_path)
imageset.load()

# get camera parameters
camera = Camera(camera_param_path)

# get bundle parameters
bundle = TagBundle()
bundle.load(bundle_param_path)

# create detector
detector_options = apriltag.DetectorOptions(families=bundle.tag_family,
                                            border=1,
                                            nthreads=4,
                                            quad_decimate=4,
                                            quad_blur=0.0,
                                            refine_edges=True,
                                            refine_decode=False,
                                            refine_pose=False,
                                            debug=False,
                                            quad_contours=True)
detector = apriltag.Detector(detector_options)

figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')
figure_2 = plt.figure()
ax_2 = figure_2.add_subplot(111, projection='3d')

draw_camera(ax, np.eye(4))
draw_axes(ax_2, np.eye(4), 0.02)

print("processing images")
for img in tqdm(imageset.images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    pnp = TagPnP()
    pnp.add_tag(detections, bundle)
    ret, rvecs, tvecs = pnp.solve(camera)

    pose = Rtvec2HomogeousT(rvecs, tvecs)
    pose_inv = np.linalg.inv(pose)

    draw_axes(ax, pose, 0.02)
    draw_camera(ax_2, pose_inv, 0.05, 0.2)

plt.show()

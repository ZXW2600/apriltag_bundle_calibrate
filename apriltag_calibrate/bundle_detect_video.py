import apriltag

import cv2
import numpy as np
import argparse
import os
import yaml

# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to the video file")
ap.add_argument("-o", "--output", required=True,
                help="path to the result path")
ap.add_argument("-c", "--camera", required=True,
                help="path to the camera calibration file")
ap.add_argument("-b", "--bundle", required=True,
                help="path to the bundle calibration file")


args = ap.parse_args()
video_path = args.video
output_path = args.output
camera_param_path = args.camera
bundle_param_path = args.bundle

# read the video
cap = cv2.VideoCapture(video_path)

# get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)




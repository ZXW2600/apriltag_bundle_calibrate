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

# setup detector
detect_option = apriltag.DetectorOptions(families='tag25h9',
                                         border=1,
                                         nthreads=1,
                                         quad_decimate=8,
                                         quad_blur=0.0,
                                         refine_edges=True,
                                         refine_decode=False,
                                         refine_pose=False,
                                         debug=False,
                                         quad_contours=True)

detector = apriltag.Detector(detect_option)

# read camera calibration

# read camera parameters
with open(camera_param_path, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
    cx = data["cx"]
    cy = data["cy"]
    fx = data["fx"]
    fy = data["fy"]
    distCoeffs = np.array(data["distCoeffs"])
    cameraMatrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
print(f"Camera parameters are loaded from {camera_param_path}"
      f"\nfx={fx}, fy={fy}, cx={cx}, cy={cy}, distCoeffs={distCoeffs}")

# read bundle calibration
bundle_pose_dict = {}
with open(bundle_param_path, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

    # get tag pose
    index_tag = 0
    while True:
        if f"t{index_tag}" in data:
            pose = np.array(data[f"t{index_tag}"])
            bundle_pose_dict[index_tag] = pose
            index_tag += 1
        else:
            break
# setup gtsam graph
    # init gtsam graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# process image
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    if len(results) > 0:
        data = {"timestamp": cap.get(
            cv2.CAP_PROP_POS_MSEC)*1e-3, "image": gray, "detections": results}
        data_list.append(data)


for result in results:
    result: apriltag.Detection
    i = 0
    for c in result.corners:
        c: tuple
        cv2.circle(img_show, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
        cv2.putText(img_show, f"{i}", (int(c[0]), int(
            c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        i += 1
    print(f"tag id:{result.tag_id}")




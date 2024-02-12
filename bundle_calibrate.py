import yaml
import numpy as np
import argparse
import apriltag
import os
import gtsam

import cv2


class KeyNote:
    CAMERA = "x"
    MASTER_TAG = "t"
    AID_TAG = "a"


# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", required=True,
                help="path to the camera calibrate file ")
ap.add_argument("-i", "--image", required=True,
                help="folder path to the input image")
ap.add_argument("-o", "--output", required=False,
                help="output file name", default="bundle_calib.yaml")
ap.add_argument("-t", "--tag", required=True,
                help="tag family")
ap.add_argument("-s", "--size", required=True,
                help="tag size")
ap.add_argument("-aid_tag", "--aid_tag", required=False,
                help="aid tag family")
ap.add_argument("-aid_size", "--aid_size", required=False,
                help="aid tag size")
# ap.add_argument("-e", "--extend", required=False,
#                 help="last calibrate result, add to graph as prior factor", default="")

camera_param_file = ap.parse_args().camera
folder_path = ap.parse_args().image
yaml_file = ap.parse_args().output

master_tag_family = ap.parse_args().tag
master_tag_size = float(ap.parse_args().size)

aid_tag_family = ap.parse_args().aid_tag
if aid_tag_family is not None:
    aid_tag_size = float(ap.parse_args().aid_size)
# last_calib_file = ap.parse_args().extend


# read camera parameters
with open(camera_param_file, 'r') as stream:
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
print(f"Camera parameters are loaded from {camera_param_file}"
      f"\nfx={fx}, fy={fy}, cx={cx}, cy={cy}, distCoeffs={distCoeffs}")


# init gtsam graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# setting detector
master_options = apriltag.DetectorOptions(families=master_tag_family,
                                          border=1,
                                          nthreads=4,
                                          quad_decimate=4,
                                          quad_blur=0.0,
                                          refine_edges=True,
                                          refine_decode=False,
                                          refine_pose=False,
                                          debug=False,
                                          quad_contours=True)
master_detector = apriltag.Detector(master_options)
master_obj_pts = np.array([[-master_tag_size / 2, -master_tag_size / 2, 0],
                           [master_tag_size / 2, -master_tag_size / 2, 0],
                           [master_tag_size / 2, master_tag_size / 2, 0],
                           [-master_tag_size / 2, master_tag_size / 2, 0]])
if aid_tag_family is not None:
    aid_options = apriltag.DetectorOptions(families=aid_tag_family,
                                           border=1,
                                           nthreads=4,
                                           quad_decimate=4,
                                           quad_blur=0.0,
                                           refine_edges=True,
                                           refine_decode=False,
                                           refine_pose=False,
                                           debug=False,
                                           quad_contours=True)
    aid_detector = apriltag.Detector(aid_options)
    aid_obj_pts = np.array([[-aid_tag_size / 2, -aid_tag_size / 2, 0],
                            [aid_tag_size / 2, -aid_tag_size / 2, 0],
                            [aid_tag_size / 2, aid_tag_size / 2, 0],
                            [-aid_tag_size / 2, aid_tag_size / 2, 0]])


# read all the image to a list
images = []

sub_dirs = [x[0] for x in os.walk(folder_path)]
for sub_dir in sub_dirs:
    bundle_images = []
    files = os.listdir(sub_dir)
    print(f"Processing {sub_dir} : {len(files)} images found.")
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(sub_dir, file))
            bundle_images.append(img)
    images.append(bundle_images)

img_index = 0
# use first image as the reference
camera_key = gtsam.symbol(KeyNote.CAMERA, img_index)

pose_raw = {}
obj_points_buffer = [master_obj_pts]
if aid_tag_family is not None:
    obj_points_buffer.append(aid_obj_pts)

bundle_index = 0
for bundle in images:
    bundle_index += 1
    for img in bundle:
        tag_pose = {}

        camera_key = gtsam.symbol(KeyNote.CAMERA, img_index)
        if not initial_estimate.exists(camera_key):
            initial_estimate.insert(camera_key, gtsam.Pose3())

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result_buffer = [master_detector.detect(gray, False)]
        symbol_buffer = [KeyNote.MASTER_TAG]

        if aid_tag_family is not None:
            aid_results = aid_detector.detect(gray, False)
            result_buffer.append(aid_detector.detect(gray, False))
            symbol_buffer.append(f"{KeyNote.AID_TAG}")
            
        master_results = master_detector.detect(gray, False)
        
        for results, symbol, obj_pts in zip(result_buffer, symbol_buffer, obj_points_buffer):
            for r in results:
                r: apriltag.DetectionBase
                tag_id = r.tag_id*1000+bundle_index
                tag_pose_key = gtsam.symbol(symbol, tag_id)

                corner = r.corners

                # Solve for the pose of the AprilTag corner
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, corner, cameraMatrix, distCoeffs)

                transform_matrix = np.diag([1.0, 1.0, 1.0, 1.0])
                transform_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                transform_matrix[:3, 3] = tvec.flatten()
                tag_pose[tag_id] = transform_matrix.tolist()

                reprojectPts, Jac = cv2.projectPoints(
                    obj_pts, rvec, tvec, cameraMatrix, distCoeffs)

                covar_ = Jac.T @ Jac
                covar_pose = np.linalg.inv(covar_[0:6, 0:6])
                covar_diag = np.diagonal(covar_pose)
                # print(f"tag_{tag_id}covar", covar_diag)
                # Create a robust noise model
                tag_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(0.1),
                    # Adjust these values as needed
                    gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
                )
                # tag_noise = gtsam.noiseModel.Isotropic.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
                # print(f"tag_{tag_id}noise", tag_noise)
                # Add the pose of the AprilTag corner to the graph
                # gtsam.factor
                graph.push_back(
                    gtsam.BetweenFactorPose3(
                        camera_key,
                        tag_pose_key,

                        gtsam.Pose3(gtsam.Rot3.Rodrigues(rvec), tvec),
                        tag_noise)
                )
                if not initial_estimate.exists(tag_pose_key):
                    # initial_estimate.insert(tag_pose_key, gtsam.Pose3(
                    #     gtsam.Rot3.Rodrigues(rvec), tvec))
                    initial_estimate.insert(tag_pose_key, gtsam.Pose3())

            pose_raw[img_index] = tag_pose
            img_index += 1

# set fixed prior for the first image
graph.add(gtsam.PriorFactorPose3(
    gtsam.symbol(KeyNote.CAMERA, 0), gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 0.1)))

# Optimize the graph
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY")
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()

save_data = {}
img_index = 0
tag_index = 0
for key in result.keys():
    chr = gtsam.symbolChr(key)
    index = gtsam.symbolIndex(key)
    if chr == ord(KeyNote.CAMERA):
        img_pose = result.atPose3(gtsam.symbol(
            KeyNote.CAMERA, index)).matrix().tolist()
        save_data[f"{KeyNote.CAMERA}{index}"] = img_pose
    elif chr == ord(KeyNote.MASTER_TAG):
        tag_pose = result.atPose3(gtsam.symbol(
            KeyNote.MASTER_TAG, index)).matrix().tolist()
        save_data[f"{KeyNote.MASTER_TAG}{index}"] = tag_pose
    elif chr == ord(KeyNote.AID_TAG):
        tag_pose = result.atPose3(gtsam.symbol(
            KeyNote.AID_TAG, index)).matrix().tolist()
        save_data[f"{KeyNote.AID_TAG}{index}"] = tag_pose
    else:
        print(f"Unknown key {chr:02x}")


graph.saveGraph("bundle_calib.dot", result)
with open(yaml_file, 'w') as file:
    yaml.dump(save_data, file)

print(f"Bundle calibration result is saved to {yaml_file}")


# save raw data
with open("bundle_calib_raw.yaml", 'w') as file:
    yaml.dump(pose_raw, file)

import yaml
import numpy as np
import argparse
import apriltag
import os
import gtsam
import gtsam_unstable
import cv2
import os, sys
ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")
    
    
from pyvis.network import Network


class KeyType:
    CAMERA = "x"
    MASTER_TAG = "t"
    AID_TAG = "a"
    POINTS = "p"


class PointsIndexGenerator:
    name_list = [
        "points_id",
        "key_id",
        "points_id"
    ]
    word_len = [
        2,
        8,
        16
    ]
    mask_list = []

    def __init__(self) -> None:
        self.shift_list = [0]
        for i in range(1, len(self.word_len)):
            self.shift_list.append(self.shift_list[i-1]+self.word_len[i-1])
        for i in range(len(self.word_len)):
            self.mask_list.append(2**self.word_len[i]-1)
    # Points Name Rules:

    def get_points_symbol(self, key_note, tag_id, points_id):
        key_id = ord(key_note)
        int_list = [points_id, key_id, tag_id]
        symbol_id = 0
        for i in range(3):
            symbol_id |= int_list[i] << self.shift_list[i]
        return gtsam.symbol(KeyType.POINTS, symbol_id)

    def resolve_points_symbol_id(self, symbol):
        symbol_index = gtsam.symbolIndex(symbol)
        int_list = []
        for i in range(3):
            int_list.append(
                (symbol_index >> self.shift_list[i]) & self.mask_list[i])
        points_id = int_list[0]
        key_id = int_list[1]
        tag_id = int_list[2]
        return tag_id, key_id, points_id

    def get_label(self, symbol):
        tag_id, key_id, points_id = self.resolve_points_symbol_id(symbol)
        return f"{key_id:c} tag {tag_id} {points_id} th point"

    def to_string(self, symbol):
        return f"{gtsam.symbolChr(symbol):c}{gtsam.symbolIndex(symbol)}"
point_index_gen = PointsIndexGenerator()


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
master_obj_pts = [np.array([-master_tag_size / 2, -master_tag_size / 2, 0]),
                  np.array([master_tag_size / 2, -master_tag_size / 2, 0]),
                  np.array([master_tag_size / 2, master_tag_size / 2, 0]),
                  np.array([-master_tag_size / 2, master_tag_size / 2, 0])]
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
    aid_obj_pts = [np.array([-aid_tag_size / 2, -aid_tag_size / 2, 0]),
                   np.array([aid_tag_size / 2, -aid_tag_size / 2, 0]),
                   np.array([aid_tag_size / 2, aid_tag_size / 2, 0]),
                   np.array([-aid_tag_size / 2, aid_tag_size / 2, 0])]


class PoseGraph:

    def __init__(self) -> None:
        # setup vis network
        self.vis = Network(select_menu=True,)

        # as a flag to check if the tag points is initialized
        self.tag_init_set: dict[set] = {}
        self.tag_init_set[KeyType.MASTER_TAG] = set()
        self.tag_init_set[KeyType.AID_TAG] = set()

        # init gtsam graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.k = gtsam.Cal3_S2(fx, fy, 0, cx, cy)

        self.params = gtsam.LevenbergMarquardtParams()
        self.params.setVerbosityLM("SUMMARY")

    def add_tag_points(self, key_type, tag_id, obj_points,
                       image_points, camera_key, tag_pose_key):
        tag_pose_key = gtsam.symbol(key_type, tag_id)
        tag_points_keys = []
        for i in range(4):
            tag_points_keys.append(
                point_index_gen.get_points_symbol(key_type, tag_id, i))

        # add points to graph
        if not self.tag_init_set[key_type].__contains__(tag_id):
            self.tag_init_set[key_type].add(tag_id)
            
            self.initial_estimate.insert(
                tag_pose_key, gtsam.Pose3())
            self.vis.add_node(
                point_index_gen.to_string(tag_pose_key), label=f"{key_type}{tag_id} pose")
            
            for i in range(4):
                self.initial_estimate.insert(
                    tag_points_keys[i], gtsam.Point3(obj_points[i]))
                self.vis.add_node(
                    point_index_gen.to_string(tag_points_keys[i]), label=point_index_gen.get_label(tag_points_keys[i]))
                
                self.graph.push_back(
                    gtsam_unstable.Pose3ToPoint3Factor(
                        tag_pose_key, tag_points_keys[i], obj_points[i],
                        gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(0.1),
                                                       gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
                    ))
                self.vis.add_edge(
                    point_index_gen.to_string(tag_pose_key), point_index_gen.to_string(tag_points_keys[i]), label=f"Pose3ToPoint3Factor"
                )

                
            

        for i in range(4):
            self.graph.push_back(
                gtsam.GenericProjectionFactorCal3_S2(
                    image_points[i].reshape(
                        2), gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
                    camera_key, tag_points_keys[i], self.k
                )
            )
            self.vis.add_edge(
                point_index_gen.to_string(camera_key), point_index_gen.to_string(tag_points_keys[i]), label=f"GenericProjectionFactorCal3_S2"
            )

    def add_cam_pose(self, camera_key):
        self.initial_estimate.insert(
            camera_key, gtsam.Pose3())
        self.vis.add_node(
            point_index_gen.to_string(camera_key), label=f"{KeyType.CAMERA}{gtsam.symbolIndex(camera_key)} pose")

    def fix_first_camera(self):
        pass

    def fix_first_tag(self):
        for key in self.graph.keys():
            chr = gtsam.symbolChr(key)
            index = gtsam.symbolIndex(key)
            if chr == ord(KeyType.MASTER_TAG):
                self.graph.add(gtsam.PriorFactorPose3(
                    key, gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 0.1)))

    def print_init(self):
        for key in self.initial_estimate.keys():
            if gtsam.symbolChr(key) == ord(KeyType.POINTS):
                tag_id, key_id, points_id = point_index_gen.resolve_points_symbol_id(
                    key)
                print(f" {key_id:c} tag {tag_id} {points_id} th point")
            else:
                print(f"{gtsam.symbolChr(key):c} {gtsam.symbolIndex(key)} ")

    def solve(self):
        # Optimize the graph

        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, self.params)
        return optimizer.optimize()

    def save_bundle(self, yaml_file):
        save_data = {}
        for key in result.keys():
            chr = gtsam.symbolChr(key)
            index = gtsam.symbolIndex(key)
            if chr == ord(KeyType.CAMERA):
                img_pose = result.atPose3(gtsam.symbol(
                    KeyType.CAMERA, index)).matrix().tolist()
                save_data[f"{KeyType.CAMERA}{index}"] = img_pose
            elif chr == ord(KeyType.MASTER_TAG):
                tag_pose = result.atPose3(gtsam.symbol(
                    KeyType.MASTER_TAG, index)).matrix().tolist()
                save_data[f"{KeyType.MASTER_TAG}{index}"] = tag_pose
            elif chr == ord(KeyType.AID_TAG):
                tag_pose = result.atPose3(gtsam.symbol(
                    KeyType.AID_TAG, index)).matrix().tolist()
                save_data[f"{KeyType.AID_TAG}{index}"] = tag_pose
            elif chr == ord(KeyType.POINTS):
                pass
            else:
                print(f"Unknown key {chr:c}")

        # graph.saveGraph("bundle_calib.dot", result)
        with open(yaml_file, 'w') as file:
            yaml.dump(save_data, file)

    def save_graph(self, dot_file):
        self.graph.saveGraph(dot_file, self.initial_estimate)

    def visualize(self):
        # self.vis.toggle_physics(True)
        self.vis.show("bundle_calib.html",local=True,notebook=False)

pose_graph = PoseGraph()

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
camera_key = gtsam.symbol(KeyType.CAMERA, img_index)

pose_raw = {}
obj_points_buffer = [master_obj_pts]
if aid_tag_family is not None:
    obj_points_buffer.append(aid_obj_pts)

bundle_index = 0
for bundle in images:
    bundle_index += 1
    for img in bundle:
        tag_pose = {}
        camera_key = gtsam.symbol(KeyType.CAMERA, img_index)
        pose_graph.add_cam_pose(camera_key)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result_buffer = [master_detector.detect(gray, False)]
        symbol_buffer = [KeyType.MASTER_TAG]

        if aid_tag_family is not None:
            aid_results = aid_detector.detect(gray, False)
            result_buffer.append(aid_detector.detect(gray, False))
            symbol_buffer.append(f"{KeyType.AID_TAG}")

        master_results = master_detector.detect(gray, False)

        for results, symbol, obj_pts in zip(result_buffer, symbol_buffer, obj_points_buffer):
            for r in results:
                r: apriltag.DetectionBase
                if symbol == KeyType.MASTER_TAG:
                    tag_id = r.tag_id
                else:
                    tag_id = r.tag_id*1000+bundle_index
                tag_pose_key = gtsam.symbol(symbol, tag_id)

                corners = r.corners

                # # undisort the corner points
                # corners = cv2.undistortPoints(
                #     corners, cameraMatrix, distCoeffs)

                pose_graph.add_tag_points(
                    symbol, tag_id, obj_pts,
                    corners, camera_key, tag_pose_key)

        pose_raw[img_index] = tag_pose
        img_index += 1

# set fixed prior for the first image
# pose_graph.fix_first_tag()

pose_graph.print_init()

pose_graph.visualize()

result = pose_graph.solve()

pose_graph.save_graph("bundle_calib.dot")
pose_graph.save_bundle(yaml_file)
print(f"Bundle calibration result is saved to {yaml_file}")

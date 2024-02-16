import symforce  # noqa
symforce.set_epsilon_to_symbol()  # noqa
from calibrate_factors import prior_pose_residual, prior_bundle_pose_residual, fixed_pose_residual, prior_bewteen_pose_residual

import sys
import cv2
from symforce.opt.optimizer import Optimizer
from symforce.opt.factor import Factor
from symforce.values import Values
import symforce.opt.noise_models as noise
import symforce.symbolic as sf
from pyvis.network import Network
import yaml
import numpy as np
import argparse
import apriltag
import os


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


class CommandLineParams:
    def __init__(self) -> None:
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
        ap.add_argument("-at", "--aid_tag", required=False,
                        help="aid tag family")
        ap.add_argument("-as", "--aid_size", required=False,
                        help="aid tag size")
        ap.add_argument("-ac", "--aid_tag_config",
                        required=False, help="aid tag config file")
        ap.add_argument("-d", "--draw", required=False,
                        help="path to save detect result",)
        # ap.add_argument("-e", "--extend", required=False,
        #                 help="last calibrate result, add to graph as prior factor", default="")
        ap.add_argument("-dw", "--draw_width", required=False, default=5)

        self.camera_param_file = ap.parse_args().camera
        self.folder_path = ap.parse_args().image
        self.yaml_file = ap.parse_args().output
        self.master_tag_family = ap.parse_args().tag
        self.master_tag_size = float(ap.parse_args().size)

        self.draw_path = ap.parse_args().draw
        self.draw_width = int(ap.parse_args().draw_width)
        if self.draw_path is not None:
            self.draw = True
            if not os.path.exists(self.draw_path):
                os.mkdir(self.draw_path)
        else:
            self.draw = False

        self.aid_tag_family = ap.parse_args().aid_tag
        if self.aid_tag_family is not None:
            self.use_aid_tag = True
            self.aid_tag_size = float(ap.parse_args().aid_size)
            self.aid_tag_config = ap.parse_args().aid_tag_config
            if self.aid_tag_config is not None:
                self.use_aid_tag_config = True
        # last_calib_file = ap.parse_args().extend


class Camera:
    def __init__(self, camera_param_file) -> None:
        # read camera parameters
        with open(camera_param_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            self.cx = data["cx"]
            self.cy = data["cy"]
            self.fx = data["fx"]
            self.fy = data["fy"]
            self.distCoeffs = np.array(data["distCoeffs"])
            self.cameraMatrix = np.array(
                [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3)
        print(f"Camera parameters are loaded from {camera_param_file}"
              f"\nfx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, distCoeffs={self.distCoeffs}")


class ApriltagDetector:
    def __init__(self, params: CommandLineParams) -> None:
        self.detector_list = []
        self.obj_points_list = []
        self.symbol_list = []

        master_tag_family = params.master_tag_family
        master_tag_size = params.master_tag_size
        aid_tag_family = params.aid_tag_family
        if aid_tag_family is not None:
            aid_tag_size = params.aid_tag_size

        # setting detector
        print("setup master tag detector")
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
                          np.array([master_tag_size / 2, -
                                    master_tag_size / 2, 0]),
                          np.array(
            [master_tag_size / 2, master_tag_size / 2, 0]),
            np.array([-master_tag_size / 2, master_tag_size / 2, 0])]
        # add master tag detector
        self.detector_list.append(master_detector)
        self.obj_points_list.append(master_obj_pts)
        self.symbol_list.append(KeyType.MASTER_TAG)

        if aid_tag_family is not None:
            print("setup aid tag detector")
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
            self.aid_detector = apriltag.Detector(aid_options)
            self.aid_obj_pts = [np.array([-aid_tag_size / 2, -aid_tag_size / 2, 0]),
                                np.array(
                                    [aid_tag_size / 2, -aid_tag_size / 2, 0]),
                                np.array(
                                    [aid_tag_size / 2, aid_tag_size / 2, 0]),
                                np.array([-aid_tag_size / 2, aid_tag_size / 2, 0])]

            # add aid tag detector
            self.detector_list.append(self.aid_detector)
            self.obj_points_list.append(self.aid_obj_pts)
            self.symbol_list.append(KeyType.AID_TAG)

        self.n_detector = len(self.detector_list)

        # print info
        print(f"master tag {master_tag_family} enabled")
        print(f"aid tag {aid_tag_family} enabled")
        print(f"detector is ready :{self.n_detector}")

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = []

        for i in range(self.n_detector):
            detector = self.detector_list[i]
            result = detector.detect(gray)
            results.append(result)
            # print(f"detect {i}", len(result), self.symbol_list[i])
        return results, self.symbol_list, self.obj_points_list


class PoseGraph:
    def __init__(self) -> None:
        self.factors = []
        self.optimize_key = []
        self.initial_estimate = Values(
            epsilon=sf.numeric_epsilon,
            camera_pose=Values(),
            bundle_pose=Values(),
            tag_pose=Values(),
            measurement=Values(),
            tag_sigmas=0.1
        )
        self.pose_key = ["camera_pose", "bundle_pose", "tag_pose"]

        self.tag_list = {}

        self.bundle_index = 0
        self.camera_index = 0

        self.graph_skip_node_list=[
            "epsilon",
            "tag_sigmas",
        ]
        self.debug_graph = Network(filter_menu=True)
        self.debug_graph.add_node("epsilon", label="epsilon")
        self.debug_graph.add_node("tag_sigmas", label="tag_sigmas")
        self.debug_graph.show_buttons(filter_=['physics'])

    def add_camera(self):
        self.camera_index += 1
        camera_pose_key = self.get_camera_key(self.camera_index)
        self.initial_estimate[camera_pose_key] = sf.Pose3()
        self.optimize_key.append(camera_pose_key)

        self.debug_graph.add_node(camera_pose_key)
        return self.camera_index

    def add_bundle(self):
        self.bundle_index += 1
        bundle_pose_key = self.get_bundle_key(self.bundle_index)
        self.initial_estimate[bundle_pose_key] = sf.Pose3()
        self.debug_graph.add_node(bundle_pose_key)

        # self.optimize_key.append(bundle_pose_key)
        return self.bundle_index

    def fix_bundle(self, bundle_id, bundle_pose):
        bundle_key = self.get_bundle_key(bundle_id)
        # add a fix factor to the graph to fix the bundle pose
        measurement_key = f"measurement.fix.bundle{bundle_key}"
        self.initial_estimate[measurement_key] = bundle_pose
        self.add_factor(residual=fixed_pose_residual,
                        keys=[bundle_key, measurement_key, "epsilon"])

    def fix_tag(self,  tag_key, tag_pose):
        # add a fix factor to the graph to fix the tag pose

        measurement_key = f"measurement.fix.tag_{tag_key}"
        self.add_measurement(measurement_key, tag_pose)

        self.add_factor(
            residual=fixed_pose_residual,
            keys=[tag_key, measurement_key, "epsilon"]
        )

    def solve(self, init_value=None):
        if init_value is not None:
            self.initial_estimate = init_value
        optimizer = Optimizer(
            factors=self.factors,
            optimized_keys=self.optimize_key,
            # So that we save more information about each iteration, to visualize later:
            debug_stats=True,
        )
        self.result = optimizer.optimize(self.initial_estimate)
        return self.result

    def save_result(self, file_name):
        save_data = {}

        for key in self.result.optimized_values.keys():
            key_data = {}
            if key in self.pose_key:
                for key_inner in self.result.optimized_values[key].keys():
                    opt_pose: sf.Pose3 = self.result.optimized_values[key][key_inner]
                    pose = sf.Pose3.from_storage(opt_pose.to_storage())
                    key_data[key_inner] = pose.to_homogenous_matrix(
                    ).to_numpy().tolist()
            save_data[key] = key_data

        with open(file_name, 'w') as file:
            yaml.dump(save_data, file)

    def get_tag_key(self, tag_type, tag_id):
        return f"tag_pose.{tag_type}{tag_id}"

    def get_bundle_key(self, bundle_id):
        return f"bundle_pose.bundle_{bundle_id}"

    def get_camera_key(self, camera_id):
        return f"camera_pose.camera_{camera_id}"

    def add_factor(self, residual, keys):
        self.factors.append(Factor(
            residual=residual,
            keys=keys
        ))
        residual_name = f"{residual.__name__}"
        residual_id=residual_name
        for key in keys:
            residual_id += f".{key}"
        self.debug_graph.add_node(residual_id, label=residual_name)
        for key in keys:
            if key in self.graph_skip_node_list:
                continue
            self.debug_graph.add_edge(key, residual_id,title=residual_name)

    def add_tag_to_graph(self, tag_type, tag_id):
        tag_name = self.get_tag_key(tag_type, tag_id)
        self.debug_graph.add_node(tag_name)

    def save_debug_graph(self,path):
        self.debug_graph.save_graph(path)
        
    def add_measurement(self, measurement_key, measurement):
        self.initial_estimate[measurement_key] = measurement
        self.debug_graph.add_node(measurement_key)


class WarmupPoseGraph(PoseGraph):

    def add_tag(self, bundle_id, camera_id, tag_type, tag_id,  rvec, tvec):

        bundle_key = self.get_bundle_key(bundle_id)
        camera_key = self.get_camera_key(camera_id)

        if tag_type in self.tag_list.keys():
            self.tag_list[tag_type].append(tag_id)
        else:
            self.tag_list[tag_type] = [tag_id]

        tag_key = self.get_tag_key(tag_type, tag_id)

        if tag_key not in self.initial_estimate:
            self.initial_estimate[tag_key] = sf.Pose3()
            self.optimize_key.append(tag_key)
            self.add_tag_to_graph(tag_type, tag_id)

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        prior_pose = sf.Pose3(
            R=sf.Rot3.from_rotation_matrix(sf.M33(rotation_matrix)), t=sf.V3(tvec))

        # tag pose in camera camera_T_tag
        mearument_key = f"measurement.camera{camera_id}.bundle_{bundle_id}.{tag_type}_{tag_id}"
        # world_T_tag_pose =world_T_bundle*bundle_T_tag
        # camera_T_tag= camera_T_world*world_T_tag
        self.add_measurement(mearument_key, prior_pose)
        # self.initial_estimate[mearument_key] = prior_pose
        # self.factors.append(Factor(
        #     residual=prior_bundle_pose_residual,
        #     keys=[camera_key,bundle_key, tag_key, mearument_key, "epsilon"],
        # ))
        self.add_factor(
                residual=prior_bewteen_pose_residual,
                keys=[camera_key, tag_key, mearument_key,
                      "epsilon", "tag_sigmas"]
            
        )
        print("add tag factor")
        return tag_key

    def fix_first_tag(self):
        for tag_type, tag_ids in self.tag_list.items():
            tag_list = sorted(tag_ids)
            tag_key = self.get_tag_key(tag_type, tag_list[0])
            self.fix_tag(tag_key, sf.Pose3())
            print(f"fix tag {tag_key} to {sf.Pose3()}")


class BundleCalibratePoseGraph(PoseGraph):
    def __init__(self, camera: Camera) -> None:
        super().__init__()

    def add_tag(self, camera_key, tag_type, tag_id, corners, tag_obj_pts):
        # add tag_pose node
        tag_pose_key = gtsam.symbol(tag_type, tag_id)
        if not self.initial_estimate.exists(tag_pose_key):
            # initial_estimate.insert(tag_pose_key, gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(rvec), tvec))
            self.initial_estimate.insert(tag_pose_key, gtsam.Pose3())

        tag_points_key = []
        for i in range(4):
            tag_points_key_i = self.point_index_generator.get_points_symbol(
                tag_type, tag_id, i)
            tag_points_key.append(tag_points_key_i)
            if not self.initial_estimate.exists(tag_points_key_i):
                self.initial_estimate.insert(
                    tag_points_key_i, gtsam.Point3(tag_obj_pts[i]))
                # add tag points edge
                self.graph.push_back(
                    gtsam_unstable.Pose3ToPoint3Factor(
                        tag_pose_key, tag_points_key_i,
                        tag_obj_pts[i], gtsam.noiseModel.Constrained.All(3))
                )

        for i in range(4):
            self.graph.push_back(
                gtsam.GenericProjectionFactorCal3_S2(
                    corners[i].reshape(
                        2), self.tag_noise[tag_type],
                    camera_key, tag_points_key[i], self.k
                )
            )


class ImageLoader:
    def __init__(self, path) -> None:
        self.path = path
        self.images = []

    def load(self):
        sub_dirs = [x[0] for x in os.walk(self.path)]
        for sub_dir in sub_dirs:
            bundle_images = []
            files = os.listdir(sub_dir)
            print(f"Searching in {sub_dir} ", end=":")
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    img = cv2.imread(os.path.join(sub_dir, file))
                    bundle_images.append((file.split("/")[-1], img))
            self.images.append(bundle_images)
            print(f"{len(bundle_images)} images loaded in {sub_dir} ")


def solve_pnp(obj_points, img_points, camera: Camera):
    ret, rvec, tvec = cv2.solvePnP(
        np.array(obj_points), np.array(img_points), camera.cameraMatrix, camera.distCoeffs)

    return rvec, tvec


def main():
    cmd_params = CommandLineParams()
    camera = Camera(cmd_params.camera_param_file)
    tag_detector = ApriltagDetector(cmd_params)
    warmup_graph = WarmupPoseGraph()
    calibr_graph = BundleCalibratePoseGraph(camera)
    image_loader = ImageLoader(cmd_params.folder_path)
    image_loader.load()

    tag_cnt = {
        KeyType.MASTER_TAG: 0,
        KeyType.AID_TAG: 0
    }
    master_bundle_key = warmup_graph.add_bundle()
    # warmup_graph.fix_bundle(master_bundle_key, sf.Pose3())

    calibr_graph.add_bundle()

    for bundle in image_loader.images:
        bundle_key = warmup_graph.add_bundle()
        calibr_graph.add_bundle()
        # add a bundle pose node
        for file, image in bundle:

            camera_key = warmup_graph.add_camera()
            calibr_graph.add_camera()
            if cmd_params.draw:
                img_show = image.copy()

            result = tag_detector.detect(image)
            for tag_results, tag_type, obj_pts in zip(*result):
                for tag in tag_results:
                    tag_cnt[tag_type] += 1

                    tag: apriltag.Detection
                    tag_id = tag.tag_id

                    if tag_type == KeyType.AID_TAG:
                        tag_bundle = bundle_key
                    else:
                        tag_bundle = master_bundle_key
                    corners = tag.corners

                    # print(corners)
                    if cmd_params.draw:
                        for p in corners:
                            cv2.circle(img_show, (int(p[0]), int(
                                p[1])), cmd_params.draw_width, (0, 50, 255), int(cmd_params.draw_width/3)+1)
                    rvec, tvec = solve_pnp(
                        obj_pts, corners, camera
                    )

                    warmup_graph.add_tag(
                        tag_bundle, camera_key, tag_type, tag_id,  rvec, tvec)
                    # calibr_graph.add_tag(
                    #     camera_key, tag_type, tag_id, corners, obj_pts)

            if cmd_params.draw:

                path = os.path.join(cmd_params.draw_path, file)
                cv2.imwrite(path, img_show)

    # solve warmup graph
    warmup_graph.fix_first_tag()
    # print(warmup_graph.optimize_key)
    warmup_graph.save_debug_graph("warmup_graph.html")
    warmup_result = warmup_graph.solve()

    # solve bundle adjustment graph
    # bundle_result = calibr_graph.solve(init_value=warmup_result)

    # summary result
    print(f"find {tag_cnt[KeyType.MASTER_TAG]} master tags")
    print(f"find {tag_cnt[KeyType.AID_TAG]} aid tags")
    # save result
    warmup_graph.save_result("warmup_result.yaml")
    # calibr_graph.save_result("bundle_result.yaml")


if __name__ == "__main__":
    main()

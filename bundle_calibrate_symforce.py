import symforce  # noqa
symforce.set_epsilon_to_symbol()  # noqa
from calibrate_factors import prior_pose_residual, prior_bundle_pose_residual, fixed_pose_residual, prior_bewteen_pose_residual, identity_pose_residual
from symforce.opt.optimizer import Optimizer
from symforce.opt.factor import Factor
from symforce.values import Values
import symforce.symbolic as sf

from pyvis.network import Network
import yaml
import numpy as np
import argparse
import apriltag
import os


import cv2
import os
import sys
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


def symforce_symbol(chr, index):
    return f"{chr}{index}"


def symforce_symbolIndex(symbol):
    return int(symbol[1:])


def symforce_symbolChr(symbol):
    return symbol[0]


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
        return symforce_symbol(KeyType.POINTS, symbol_id)

    def resolve_points_symbol_id(self, symbol):
        symbol_index = symforce_symbolIndex(symbol)
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
        return f"{symforce_symbolChr(symbol):c}{symforce_symbolIndex(symbol)}"


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
            print(f"detect {i}", len(result), self.symbol_list[i])
        return results, self.symbol_list, self.obj_points_list


class PoseGraph:
    def __init__(self) -> None:
        self.factors = []
        self.initial_estimate = Values(
            epsilon=sf.numeric_epsilon
        )
        self.optimize_key = []
        self.point_index_generator = PointsIndexGenerator()

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

        for key in self.result.optimized_values.keys_recursive():
            if key=="epsilon" or key[:3]=="cam":
                continue
            chr = symforce_symbolChr(key)
            index = symforce_symbolIndex(key)
            if chr == KeyType.CAMERA:
                opt_pose: sf.Pose3 = self.result.optimized_values[key]
                img_pose = sf.Pose3.from_storage(opt_pose.to_storage()).to_homogenous_matrix().to_numpy().tolist()

                save_data[f"{KeyType.CAMERA}{index}"] = img_pose

            elif chr == KeyType.MASTER_TAG:
                opt_pose: sf.Pose3 = self.result.optimized_values[key]
                tag_pose = sf.Pose3.from_storage(opt_pose.to_storage()).to_homogenous_matrix().to_numpy().tolist()
                save_data[f"{KeyType.MASTER_TAG}{index}"] = tag_pose

            elif chr == KeyType.AID_TAG:
                opt_pose: sf.Pose3 = self.result.optimized_values[key]
                tag_pose = sf.Pose3.from_storage(opt_pose.to_storage()).to_homogenous_matrix().to_numpy().tolist()
                save_data[f"{KeyType.AID_TAG}{index}"] = tag_pose

            # elif chr == ord(KeyType.POINTS):
            #     points = self.result.atPoint3(symforce_symbol(
            #         KeyType.POINTS, index)).tolist()
            #     save_data[f"{KeyType.POINTS}{index}"] = points
            else:
                print(f"Unknown key {chr}")

        with open(file_name, 'w') as file:
            yaml.dump(save_data, file)

    def fix_first_tag(self):
        for key in self.initial_estimate.keys():
            chr = symforce_symbolChr(key)
            if chr == ord(KeyType.MASTER_TAG):
                self.factors.append(
                    Factor(
                        residual=identity_pose_residual,
                        keys=[key,  "epsilon"],
                    )
                )
                return

    def fix_first_camera(self):
        pass


class WarmupPoseGraph(PoseGraph):

    def add_tag(self, camera_key, tag_type, tag_id, tag_obj_pts, rvec, tvec):

        # add tag_pose node
        tag_pose_key = symforce_symbol(tag_type, tag_id)
        if tag_pose_key not in self.initial_estimate:
            # initial_estimate.insert(tag_pose_key, gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(rvec), tvec))
            self.initial_estimate[tag_pose_key] = sf.Pose3()
            self.optimize_key.append(tag_pose_key)
            
            # self.initial_estimate.insert(tag_pose_key, gtsam.Pose3())

        add_points = False
        # add tag points node
        if add_points:
            tag_points_key = []
            for i in range(4):
                tag_points_key_i = self.point_index_generator.get_points_symbol(
                    tag_type, tag_id, i)
                tag_points_key.append(tag_points_key_i)
                if not self.initial_estimate.exists(tag_points_key_i):
                    self.initial_estimate.insert(
                        tag_points_key_i, gtsam.Point3(tag_obj_pts[i]))
                    # add tag points edge
                    self.factors.push_back(
                        gtsam_unstable.Pose3ToPoint3Factor(
                            tag_pose_key, tag_points_key_i,
                            tag_obj_pts[i], gtsam.noiseModel.Constrained.All(3))
                    )

        # tag_noise = gtsam.noiseModel.Robust.Create(
        #     gtsam.noiseModel.mEstimator.Huber.Create(0.1),
        #     # Adjust these values as needed
        #     gtsam.noiseModel.Diagonal.Sigmas(
        #         np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        # )
        # add tag pose edge
        mea_key=f"cam{camera_key}_tag{tag_pose_key}"
        R=cv2.Rodrigues(rvec)[0]
        
        self.initial_estimate[mea_key]=sf.Pose3(sf.Rot3.from_rotation_matrix(R), sf.V3(tvec))
        self.factors.append(
            Factor(
                residual=prior_bewteen_pose_residual,
                keys=[camera_key, tag_pose_key, mea_key, "epsilon"],
            )
        )

    def add_camera(self, camera_key):
        # add camera pose node
        self.initial_estimate[camera_key]=sf.Pose3()
        self.optimize_key.append(camera_key)



class BundleCalibratePoseGraph(PoseGraph):
    def __init__(self, camera: Camera) -> None:
        super().__init__()
        # self.k = gtsam.Cal3_S2(camera.fx, camera.fy, 0,
        #                        camera.cx, camera.cy,)
        # master_tag_noise = gtsam.noiseModel.Robust.Create(
        #     gtsam.noiseModel.mEstimator.Huber.Create(0.1),
        #     # Adjust these values as needed
        #     gtsam.noiseModel.Diagonal.Sigmas(
        #         np.array([2, 2]))
        # )
        # aid_tag_noise = gtsam.noiseModel.Robust.Create(
        #     gtsam.noiseModel.mEstimator.Huber.Create(0.1),
        #     # Adjust these values as needed
        #     gtsam.noiseModel.Diagonal.Sigmas(
        #         np.array([1, 1]))
        # )
        # self.tag_noise = {
        #     KeyType.MASTER_TAG: master_tag_noise,
        #     KeyType.AID_TAG: aid_tag_noise
        # }

    def add_tag(self, camera_key, tag_type, tag_id, corners, tag_obj_pts):
        # add tag_pose node
        tag_pose_key = symforce_symbol(tag_type, tag_id)
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
                self.factors.push_back(
                    gtsam_unstable.Pose3ToPoint3Factor(
                        tag_pose_key, tag_points_key_i,
                        tag_obj_pts[i], gtsam.noiseModel.Constrained.All(3))
                )

        for i in range(4):
            self.factors.push_back(
                gtsam.GenericProjectionFactorCal3_S2(
                    corners[i].reshape(
                        2), self.tag_noise[tag_type],
                    camera_key, tag_points_key[i], self.k
                )
            )

    def add_camera(self, camera_key):
        # add camera pose node
        self.initial_estimate.insert(
            camera_key, gtsam.Pose3()
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
            print(f"Processing {sub_dir} ")
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
    bundle_graph = BundleCalibratePoseGraph(camera)
    image_loader = ImageLoader(cmd_params.folder_path)
    image_loader.load()

    tag_cnt = {
        KeyType.MASTER_TAG: 0,
        KeyType.AID_TAG: 0
    }
    bundle_index = 0
    for bundle in image_loader.images:
        img_index = 0
        for file, image in bundle:
            camera_key = symforce_symbol(KeyType.CAMERA, img_index)
            if cmd_params.draw:
                img_show = image.copy()
            warmup_graph.add_camera(camera_key)
            # bundle_graph.add_camera(camera_key)

            result = tag_detector.detect(image)
            for tag_results, tag_type, obj_pts in zip(*result):
                for tag in tag_results:
                    tag_cnt[tag_type] += 1

                    tag: apriltag.Detection
                    if tag_type == KeyType.AID_TAG:
                        tag_id = bundle_index+tag.tag_id*1000
                    else:
                        tag_id = tag.tag_id
                    corners = tag.corners

                    # undistort
                    corners = cv2.undistortImagePoints(
                        corners, camera.cameraMatrix, camera.distCoeffs).reshape((4, 2))
                    # print(corners)
                    if cmd_params.draw:
                        for p in corners:
                            cv2.circle(img_show, (int(p[0]), int(
                                p[1])), cmd_params.draw_width, (0, 50, 255), int(cmd_params.draw_width/3)+1)
                    rvec, tvec = solve_pnp(
                        obj_pts, corners, camera
                    )

                    warmup_graph.add_tag(
                        camera_key, tag_type, tag_id, obj_pts, rvec, tvec)
                    # bundle_graph.add_tag(
                    #     camera_key, tag_type, tag_id, corners, obj_pts)

            if cmd_params.draw:

                path = os.path.join(cmd_params.draw_path, file)
                cv2.imwrite(path, img_show)
            # update bundle index
            img_index += 1

        # update bundle index
        bundle_index += 1
    # solve warmup graph
    warmup_graph.fix_first_tag()
    warmup_result = warmup_graph.solve()

    # solve bundle adjustment graph
    # bundle_result = bundle_graph.solve(init_value=warmup_result)

    # summary result
    print(f"find {tag_cnt[KeyType.MASTER_TAG]} master tags")
    print(f"find {tag_cnt[KeyType.AID_TAG]} aid tags")
    # save result
    warmup_graph.save_result("warmup_result.yaml")
    # bundle_graph.save_result("bundle_result.yaml")


if __name__ == "__main__":
    main()

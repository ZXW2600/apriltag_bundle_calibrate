from pyvis.network import Network
import yaml
import numpy as np
import argparse
import apriltag
import os
import gtsam
import gtsam_unstable
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
        ap.add_argument("-aid_tag", "--aid_tag", required=False,
                        help="aid tag family")
        ap.add_argument("-aid_size", "--aid_size", required=False,
                        help="aid tag size")
        # ap.add_argument("-e", "--extend", required=False,
        #                 help="last calibrate result, add to graph as prior factor", default="")

        self.camera_param_file = ap.parse_args().camera
        self.folder_path = ap.parse_args().image
        self.yaml_file = ap.parse_args().output
        self.master_tag_family = ap.parse_args().tag
        self.master_tag_size = float(ap.parse_args().size)

        self.aid_tag_family = ap.parse_args().aid_tag
        if self.aid_tag_family is not None:
            self.aid_tag_size = float(ap.parse_args().aid_size)
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

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = []
        for i in range(len(self.detector_list)):
            detector = self.detector_list[i]
            result = detector.detect(gray)
            results.append(result)
            return results, self.symbol_list, self.obj_points_list


class PoseGraph:
    def __init__(self) -> None:
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.optimize_option = gtsam.LevenbergMarquardtParams()
        self.optimize_option.setVerbosityLM("SUMMARY")

        self.point_index_generator = PointsIndexGenerator()

    def solve(self, init_value=None):
        if init_value is not None:
            self.initial_estimate = init_value
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, self.optimize_option)
        self.result = optimizer.optimize()
        return self.result

    def save_result(self, file_name):
        save_data = {}

        for key in self.result.keys():
            chr = gtsam.symbolChr(key)
            index = gtsam.symbolIndex(key)
            if chr == ord(KeyType.CAMERA):
                img_pose = self.result.atPose3(gtsam.symbol(
                    KeyType.CAMERA, index)).matrix().tolist()
                save_data[f"{KeyType.CAMERA}{index}"] = img_pose

            elif chr == ord(KeyType.MASTER_TAG):
                tag_pose = self.result.atPose3(gtsam.symbol(
                    KeyType.MASTER_TAG, index)).matrix().tolist()
                save_data[f"{KeyType.MASTER_TAG}{index}"] = tag_pose

            elif chr == ord(KeyType.AID_TAG):
                tag_pose = self.result.atPose3(gtsam.symbol(
                    KeyType.AID_TAG, index)).matrix().tolist()
                save_data[f"{KeyType.AID_TAG}{index}"] = tag_pose
            elif chr == ord(KeyType.POINTS):
                points = self.result.atPoint3(gtsam.symbol(
                    KeyType.POINTS, index)).tolist()
                save_data[f"{KeyType.POINTS}{index}"] = points
            else:
                print(f"Unknown key {chr:02x}")

        with open(file_name, 'w') as file:
            yaml.dump(save_data, file)

    def fix_first_tag(self):
        for key in self.initial_estimate.keys():
            chr = gtsam.symbolChr(key)
            if chr == ord(KeyType.MASTER_TAG):
                self.graph.add(
                    gtsam.PriorFactorPose3(
                        key, gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 1e-8)))
                return

    def fix_first_camera(self):
        pass


class WarmupPoseGraph(PoseGraph):


    def add_tag(self, camera_key, tag_type, tag_id, tag_obj_pts, rvec, tvec):

        # add tag_pose node
        tag_pose_key = gtsam.symbol(tag_type, tag_id)
        if not self.initial_estimate.exists(tag_pose_key):
            # initial_estimate.insert(tag_pose_key, gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(rvec), tvec))
            self.initial_estimate.insert(tag_pose_key, gtsam.Pose3())

        add_points = True
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
                    self.graph.push_back(
                        gtsam_unstable.Pose3ToPoint3Factor(
                            tag_pose_key, tag_points_key_i,
                            tag_obj_pts[i], gtsam.noiseModel.Constrained.All(3))
                    )

        tag_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            # Adjust these values as needed
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        )
        # add tag pose edge
        self.graph.push_back(
            gtsam.BetweenFactorPose3(
                camera_key,
                tag_pose_key,
                gtsam.Pose3(gtsam.Rot3.Rodrigues(rvec), tvec),
                tag_noise)
        )

    def add_camera(self, camera_key):
        # add camera pose node
        self.initial_estimate.insert(
            camera_key, gtsam.Pose3()
        )


class BundleCalibratePoseGraph(PoseGraph):
    def __init__(self, camera: Camera) -> None:
        super().__init__()
        self.k = gtsam.Cal3_S2(camera.fx, camera.fy, 0,
                               camera.cx, camera.cy,)
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

        tag_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            # Adjust these values as needed
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1, 1]))
        )
        for i in range(4):
            self.graph.push_back(
                gtsam.GenericProjectionFactorCal3_S2(
                    corners[i].reshape(
                        2), tag_noise,
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
                    bundle_images.append(img)
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

    bundle_index = 0
    for bundle in image_loader.images:
        img_index = 0
        for image in bundle:
            camera_key = gtsam.symbol(KeyType.CAMERA, img_index)

            warmup_graph.add_camera(camera_key)
            bundle_graph.add_camera(camera_key)

            result = tag_detector.detect(image)
            for tag_results, tag_type, obj_pts in zip(*result):
                for tag in tag_results:
                    tag: apriltag.Detection
                    if tag_type == KeyType.AID_TAG:
                        tag_id = bundle_index+tag.tag_id*1000
                    else:
                        tag_id = tag.tag_id
                    corners = tag.corners
                    rvec, tvec = solve_pnp(
                        obj_pts, corners, camera
                    )

                    warmup_graph.add_tag(
                        camera_key, tag_type, tag_id, obj_pts, rvec, tvec)
                    bundle_graph.add_tag(camera_key,tag_type, tag_id, corners,obj_pts)

            # update bundle index
            img_index += 1

        # update bundle index
        bundle_index += 1
    # solve warmup graph
    warmup_graph.fix_first_tag()
    warmup_result = warmup_graph.solve()

    # solve bundle adjustment graph
    bundle_result = bundle_graph.solve(init_value=warmup_result)

    # save result
    warmup_graph.save_result("warmup_result.yaml")
    bundle_graph.save_result("bundle_result.yaml")


if __name__ == "__main__":
    main()

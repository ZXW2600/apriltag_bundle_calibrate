from functools import partial
from tqdm import tqdm
import yaml
import numpy as np
import argparse
import apriltag
import os
import gtsam
import cv2
import os
import sys

import custom_factor
from concurrent.futures import ProcessPoolExecutor

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
    BUNDLE = "b"


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
            self.distCoeffs = np.array(data["distCoeffs"]).flatten()
            if self.distCoeffs.shape[0] > 4:
                self.k1 = self.distCoeffs[0]
                self.k2 = self.distCoeffs[1]
                self.p1 = self.distCoeffs[2]
                self.p2 = self.distCoeffs[3]

            self.cameraMatrix = np.array(
                [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3)
        print(f"Camera parameters are loaded from {camera_param_file}"
              f"\nfx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, distCoeffs={self.distCoeffs}")


class ApriltagDetector:
    def __init__(self, params: CommandLineParams) -> None:
        self.detector_list = []
        self.tag_size_list = []
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
        self.tag_size_list.append(master_tag_size)
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
            self.tag_size_list.append(aid_tag_size)

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
        return results


class PoseGraph:
    def __init__(self) -> None:
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.optimize_option = gtsam.LevenbergMarquardtParams()
        self.optimize_option.setVerbosityLM("SUMMARY")

        self.bundle_index = 0
        self.camere_index = 0

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
            elif chr == ord(KeyType.BUNDLE):
                bundle_pose = self.result.atPose3(gtsam.symbol(
                    KeyType.BUNDLE, index)).matrix().tolist()
                save_data[f"{KeyType.BUNDLE}{index}"] = bundle_pose
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

    def add_bundle(self):
        self.bundle_index += 1
        bundle_key = gtsam.symbol(KeyType.BUNDLE, self.bundle_index)
        self.initial_estimate.insert(bundle_key, gtsam.Pose3())
        return bundle_key

    def add_camera(self):
        self.camere_index += 1
        camera_key = gtsam.symbol(KeyType.CAMERA, self.camere_index)
        self.initial_estimate.insert(
            camera_key, gtsam.Pose3()
        )
        return camera_key


class WarmupPoseGraph(PoseGraph):

    def add_tag(self, bundle_key, camera_key, tag_type, tag_id, tag_obj_pts, rvec, tvec):
        # add tag_pose node
        tag_pose_key = gtsam.symbol(tag_type, tag_id)
        if not self.initial_estimate.exists(tag_pose_key):
            # initial_estimate.insert(tag_pose_key, gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(rvec), tvec))
            self.initial_estimate.insert(tag_pose_key, gtsam.Pose3())
        tag_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.3),
            # Adjust these values as needed
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        )
        self.graph.push_back(
            custom_factor.BundleCameraPnPFactor(bundle_key, tag_pose_key, camera_key,
                                                gtsam.Pose3(
                                                    gtsam.Rot3.Rodrigues(rvec), tvec),
                                                tag_noise)
        )


class BundleCalibratePoseGraph(PoseGraph):
    def __init__(self, camera: Camera) -> None:
        super().__init__()
        self.k = gtsam.Cal3DS2(fx=camera.fx, fy=camera.fy, s=0, u0=camera.cx, v0=camera.cy,
                               k1=camera.k1, k2=camera.k2, p1=camera.p1, p2=camera.p2)
        master_tag_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            # Adjust these values as needed
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([2, 2,]))
        )
        aid_tag_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(0.1),
            # Adjust these values as needed
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1, 1,]))
        )
        self.tag_noise = {
            KeyType.MASTER_TAG: master_tag_noise,
            KeyType.AID_TAG: aid_tag_noise
        }

    def add_tag(self, bundle_key, camera_key, tag_type, tag_id, corners, tag_size):
        # add tag_pose node
        tag_pose_key = gtsam.symbol(tag_type, tag_id)
        if not self.initial_estimate.exists(tag_pose_key):
            # initial_estimate.insert(tag_pose_key, gtsam.Pose3(
            #     gtsam.Rot3.Rodrigues(rvec), tvec))
            self.initial_estimate.insert(tag_pose_key, gtsam.Pose3())

        self.graph.push_back(
            custom_factor.BundleTagFactors(
                corners, self.k, tag_size, camera_key, tag_pose_key, bundle_key, self.tag_noise[
                    tag_type]
            )
        )


class ImageLoader:
    def __init__(self, path) -> None:
        self.path = path
        self.images = []

    def load_img(self, folder_path, file_path):
        if not file_path.endswith(".jpg"):
            return False, "", None
        image_path = os.path.join(folder_path, file_path)
        image = cv2.imread(image_path)
        return image is not None, file_path.split("/")[-1], image

    def load_bundle(self, folder, files):
        bundle = []
        print("reading images...")
        with ProcessPoolExecutor() as executor:
            images = list(
                tqdm(executor.map(partial(self.load_img, folder), files), total=len(files)))
            for img in images:
                if img[0]:
                    bundle.append((img[1], img[2]))
        return bundle

    def load(self):
        for folder, _, files in os.walk(self.path):
            bundle = self.load_bundle(folder, files)
            self.images.append(bundle)


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
    for bundle in image_loader.images:
        bundle_key = warmup_graph.add_bundle()
        bundle_graph.add_bundle()
        print("processing bundle...")
        for file, image in tqdm(bundle):

            if cmd_params.draw:
                img_show = image.copy()
            camera_key = warmup_graph.add_camera()
            bundle_graph.add_camera()

            result = tag_detector.detect(image)
            for tag_results, tag_type, obj_pts, tag_size in zip(result, tag_detector.symbol_list, tag_detector.obj_points_list, tag_detector.tag_size_list):
                for tag in tag_results:
                    tag_cnt[tag_type] += 1

                    tag: apriltag.Detection
                    tag_id = tag.tag_id
                    corners = tag.corners

                    if cmd_params.draw:
                        for p in corners:
                            cv2.circle(img_show, (int(p[0]), int(
                                p[1])), cmd_params.draw_width, (0, 50, 255), int(cmd_params.draw_width/3)+1)
                    rvec, tvec = solve_pnp(
                        obj_pts, corners, camera
                    )

                    warmup_graph.add_tag(
                        bundle_key, camera_key, tag_type, tag_id, obj_pts, rvec, tvec)
                    bundle_graph.add_tag(
                        bundle_key, camera_key, tag_type, tag_id, corners, tag_size)

            if cmd_params.draw:

                path = os.path.join(cmd_params.draw_path, file)
                cv2.imwrite(path, img_show)

    # solve warmup graph
    warmup_graph.fix_first_tag()
    warmup_result = warmup_graph.solve()
    warmup_graph.save_result("warmup_result.yaml")

    # solve bundle adjustment graph
    bundle_graph.fix_first_tag()
    bundle_result = bundle_graph.solve(init_value=warmup_result)

    # summary result
    print(f"find {tag_cnt[KeyType.MASTER_TAG]} master tags")
    print(f"find {tag_cnt[KeyType.AID_TAG]} aid tags")
    # save result
    bundle_graph.save_result("bundle_result.yaml")


if __name__ == "__main__":
    main()

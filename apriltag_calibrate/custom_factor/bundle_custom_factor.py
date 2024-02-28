from typing import List
import numpy as np
import gtsam
from functools import partial


def GtsamMatrix(row, col):
    return np.zeros((row, col), order='F')


def bundle_func(this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_bundle_key = this.keys()[0]
    bundle_T_body_key = this.keys()[1]
    world_T_body_key = this.keys()[2]

    world_T_bundle = v.atPose3(world_T_bundle_key)
    bundle_T_body = v.atPose3(bundle_T_body_key)
    world_T_body = v.atPose3(world_T_body_key)

    J_world_T_body_predicted_wrt_world_T_bundle = GtsamMatrix(6, 6)
    J_world_T_body_predicted_wrt_bundle_T_body = GtsamMatrix(6, 6)

    world_T_body_predicted = world_T_bundle.compose(
        bundle_T_body,
        J_world_T_body_predicted_wrt_world_T_bundle,
        J_world_T_body_predicted_wrt_bundle_T_body)
    J_error_wrt_world_T_body_predicted = GtsamMatrix(6, 6)
    J_error_wrt_world_T_body = GtsamMatrix(6, 6)
    error = world_T_body_predicted.localCoordinates(
        world_T_body,
        J_error_wrt_world_T_body_predicted,
        J_error_wrt_world_T_body)

    if H is not None:
        # error wrt world_T_bundle_key
        H[0] = J_world_T_body_predicted_wrt_world_T_bundle@J_error_wrt_world_T_body_predicted
        H[1] = J_world_T_body_predicted_wrt_bundle_T_body@J_error_wrt_world_T_body_predicted
        H[2] = J_error_wrt_world_T_body
    return error  # 1-d numpy array


def BundlePoseFactor(world_T_bundle_key: int, bundle_T_body_key: int, world_T_body_key: int, noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_bundle_key, bundle_T_body_key, world_T_body_key]
    return gtsam.CustomFactor(noise_model, keys, bundle_func)


def bundle_prior_func(world_T_body: gtsam.Pose3, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_bundle_key = this.keys()[0]
    bundle_T_body_key = this.keys()[1]

    world_T_bundle = v.atPose3(world_T_bundle_key)
    bundle_T_body = v.atPose3(bundle_T_body_key)

    J_world_T_body_predicted_wrt_world_T_bundle = GtsamMatrix(6, 6)
    J_world_T_body_predicted_wrt_bundle_T_body = GtsamMatrix(6, 6)

    world_T_body_predicted = world_T_bundle.compose(
        bundle_T_body,
        J_world_T_body_predicted_wrt_world_T_bundle,
        J_world_T_body_predicted_wrt_bundle_T_body)
    J_error_wrt_world_T_body_predicted = GtsamMatrix(6, 6)
    J_error_wrt_world_T_body = GtsamMatrix(6, 6)
    error = world_T_body_predicted.localCoordinates(
        world_T_body,
        J_error_wrt_world_T_body_predicted,
        J_error_wrt_world_T_body)

    if H is not None:
        # error wrt world_T_bundle_key
        H[0] = J_world_T_body_predicted_wrt_world_T_bundle@J_error_wrt_world_T_body_predicted
        H[1] = J_world_T_body_predicted_wrt_bundle_T_body@J_error_wrt_world_T_body_predicted
    return error  # 1-d numpy array


def BundlePosePriorFactor(world_T_bundle_key: int, bundle_T_body_key: int, world_T_body_pose: gtsam.Pose3, noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_bundle_key, bundle_T_body_key]
    return gtsam.CustomFactor(keys=keys, noiseModel=noise_model, errorFunction=partial(bundle_prior_func, world_T_body_pose))


def bundle_prior_camera_func(camera_T_body: gtsam.Pose3, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_bundle_key = this.keys()[0]
    bundle_T_body_key = this.keys()[1]
    world_T_camera_key = this.keys()[2]

    world_T_bundle = v.atPose3(world_T_bundle_key)
    bundle_T_body = v.atPose3(bundle_T_body_key)
    world_T_camera = v.atPose3(world_T_camera_key)

    J_world_T_body_wrt_world_T_bundle = GtsamMatrix(6, 6)
    J_world_T_body_wrt_bundle_T_body = GtsamMatrix(6, 6)

    world_T_body = world_T_bundle.compose(
        bundle_T_body,
        J_world_T_body_wrt_world_T_bundle,
        J_world_T_body_wrt_bundle_T_body)

    J_camera_T_body_predict_wrt_world_T_body = GtsamMatrix(6, 6)
    J_camera_T_body_predict_wrt_world_T_camera = GtsamMatrix(6, 6)
    camera_T_body_predicted = world_T_camera.between(
        world_T_body,
        J_camera_T_body_predict_wrt_world_T_camera,
        J_camera_T_body_predict_wrt_world_T_body,

    )

    J_error_wrt_camera_T_body_predict = GtsamMatrix(6, 6)
    error = camera_T_body_predicted.localCoordinates(
        camera_T_body,
        J_error_wrt_camera_T_body_predict)

    if H is not None:
        # error wrt world_T_bundle_key
        # world_T_bundle_key = this.keys()[0]
        # bundle_T_body_key = this.keys()[1]
        # world_T_camera_key = this.keys()[2]
        H[0] = J_error_wrt_camera_T_body_predict@J_camera_T_body_predict_wrt_world_T_body@J_world_T_body_wrt_world_T_bundle
        H[1] = J_error_wrt_camera_T_body_predict@J_camera_T_body_predict_wrt_world_T_body@J_world_T_body_wrt_bundle_T_body
        H[2] = J_error_wrt_camera_T_body_predict@J_camera_T_body_predict_wrt_world_T_camera
    return error  # 1-d numpy array


def BundleCameraPnPFactor(
        world_T_bundle_key: int,
        bundle_T_body_key: int,
        world_T_camera_key: int,
        camera_T_body_pose: gtsam.Pose3,
        noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_bundle_key, bundle_T_body_key, world_T_camera_key]
    return gtsam.CustomFactor(keys=keys, noiseModel=noise_model, errorFunction=partial(bundle_prior_camera_func, camera_T_body_pose))


def tag_projection_error(corners: list[np.ndarray], K: gtsam.Cal3DS2, tag_size: float, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_camera_pose_key = this.keys()[0]
    world_T_tag_pose_key = this.keys()[1]
    world_T_camera_pose = v.atPose3(world_T_camera_pose_key)
    world_T_tag_pose = v.atPose3(world_T_tag_pose_key)

    camera = gtsam.PinholePoseCal3DS2(pose=world_T_camera_pose, K=K)

    tag_T_obj_pts = [
        gtsam.Point3(-tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, tag_size / 2, 0),
        gtsam.Point3(-tag_size / 2, tag_size / 2, 0)]

    J_error_wrt_world_T_camera_pose_list = []
    J_error_wrt_world_T_tag_pose_list = []
    errors = []

    for tag_T_obj_pt, tag_uv_point in zip(tag_T_obj_pts, corners):
        J_world_T_obj_pts_wrt_world_T_tag = GtsamMatrix(3, 6)
        J_world_T_obj_pts_wrt_tag_T_points = GtsamMatrix(3, 3)

        world_T_obj_pt = world_T_tag_pose.transformFrom(
            tag_T_obj_pt, J_world_T_obj_pts_wrt_world_T_tag, J_world_T_obj_pts_wrt_tag_T_points)

        J_corner_wrt_world_T_obj_point = GtsamMatrix(2, 3)
        J_corner_wrt_camera_pose = GtsamMatrix(2, 6)
        J_world_T_obj_pts_wrt_cam_cal = GtsamMatrix(2, 9)

        try:
            corner_predicted = camera.project(
                point=world_T_obj_pt,
                Dpose=J_corner_wrt_camera_pose,
                Dpoint=J_corner_wrt_world_T_obj_point,
                Dcal=J_world_T_obj_pts_wrt_cam_cal)
            errors.append(corner_predicted - tag_uv_point)

        except:
            errors.append(np.array([1, 1])*2*K.fx())

        J_error_wrt_world_T_camera_pose_list.append(
            J_corner_wrt_camera_pose@J_corner_wrt_world_T_obj_point)
        J_error_wrt_world_T_tag_pose_list.append(
            J_corner_wrt_camera_pose@J_corner_wrt_world_T_obj_point)
    if H is not None:
        H[0] = np.column_stack(J_error_wrt_world_T_camera_pose_list)
        H[1] = np.column_stack(J_error_wrt_world_T_tag_pose_list)

    return np.concatenate(errors)


def tag_projection_error_i(index: int, corners: list[np.ndarray], K: gtsam.Cal3DS2, tag_size: float, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_camera_pose_key = this.keys()[0]
    world_T_tag_pose_key = this.keys()[1]
    world_T_camera_pose = v.atPose3(world_T_camera_pose_key)
    world_T_tag_pose = v.atPose3(world_T_tag_pose_key)

    camera = gtsam.PinholePoseCal3DS2(pose=world_T_camera_pose, K=K)

    tag_T_obj_pts = [
        gtsam.Point3(-tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, tag_size / 2, 0),
        gtsam.Point3(-tag_size / 2, tag_size / 2, 0)]

    tag_T_obj_pt, tag_uv_point = tag_T_obj_pts[index], corners[index]
    J_world_T_obj_pts_wrt_world_T_tag = GtsamMatrix(3, 6)
    J_world_T_obj_pts_wrt_tag_T_points = GtsamMatrix(3, 3)

    world_T_obj_pt = world_T_tag_pose.transformFrom(
        tag_T_obj_pt, J_world_T_obj_pts_wrt_world_T_tag, J_world_T_obj_pts_wrt_tag_T_points)

    J_corner_wrt_world_T_obj_point = GtsamMatrix(2, 3)
    J_corner_wrt_camera_pose = GtsamMatrix(2, 6)
    J_world_T_obj_pts_wrt_cam_cal = GtsamMatrix(2, 9)
    error = 0
    try:
        corner_predicted = camera.project(
            point=world_T_obj_pt,
            Dpose=J_corner_wrt_camera_pose,
            Dpoint=J_corner_wrt_world_T_obj_point,
            Dcal=J_world_T_obj_pts_wrt_cam_cal)
        error = corner_predicted - tag_uv_point

    except:
        error = np.array([1, 1])*2*K.fx()

    if H is not None:
        H[0] = J_corner_wrt_camera_pose@J_corner_wrt_world_T_obj_point
        H[1] = J_corner_wrt_camera_pose@J_corner_wrt_world_T_obj_point

    return error


def TagProjectionFactor(corners: list[np.ndarray], k: gtsam.Cal3DS2, tag_size: float, world_T_camera_pose_key: int, world_T_tag_pose_key: int, noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_camera_pose_key, world_T_tag_pose_key]
    return gtsam.CustomFactor(keys=keys, noiseModel=noise_model, error_func=partial(tag_projection_error, corners, k, tag_size))


def TagProjectionFactors(corners: list[np.ndarray], k: gtsam.Cal3DS2, tag_size: float, world_T_camera_pose_key: int, world_T_tag_pose_key: int, noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_camera_pose_key, world_T_tag_pose_key]
    factors = gtsam.NonlinearFactorGraph()
    for i in range(4):
        factors.push_back(gtsam.CustomFactor(keys=keys, noiseModel=noise_model,
                                             error_func=partial(tag_projection_error_i, i, corners, k, tag_size)))
    return factors


def bundle_projection_error(corners: list[np.ndarray], K: gtsam.Cal3DS2, tag_size: float, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_camera_pose_key = this.keys()[0]
    bundle_T_tag_pose_key = this.keys()[1]
    world_T_bundle_pose_key = this.keys()[2]
    world_T_camera_pose = v.atPose3(world_T_camera_pose_key)
    bundle_T_tag_pose = v.atPose3(bundle_T_tag_pose_key)
    world_T_bundle_pose = v.atPose3(world_T_bundle_pose_key)

    camera = gtsam.PinholePoseCal3DS2(pose=world_T_camera_pose, K=K)

    tag_T_obj_pts = [
        gtsam.Point3(-tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, tag_size / 2, 0),
        gtsam.Point3(-tag_size / 2, tag_size / 2, 0)]

    J_world_T_tag_pose_wrt_world_T_bundle_pose = GtsamMatrix(6, 6)
    J_world_T_tag_pose_wrt_bundle_T_tag_pose = GtsamMatrix(6, 6)
    world_T_tag_pose = world_T_bundle_pose.compose(
        bundle_T_tag_pose, J_world_T_tag_pose_wrt_world_T_bundle_pose, J_world_T_tag_pose_wrt_bundle_T_tag_pose)

    J_uv_wrt_world_T_camera_pose_list = []
    J_uv_wrt_bundle_T_tag_pose_list = []
    J_uv_wrt_world_T_bundle_pose_list = []

    errors = []

    for tag_T_obj_pt, tag_uv_point in zip(tag_T_obj_pts, corners):
        J_world_T_obj_pts_wrt_world_T_tag = GtsamMatrix(3, 6)
        J_world_T_obj_pts_wrt_tag_T_points = GtsamMatrix(3, 3)

        world_T_obj_pt = world_T_tag_pose.transformFrom(
            tag_T_obj_pt, J_world_T_obj_pts_wrt_world_T_tag, J_world_T_obj_pts_wrt_tag_T_points)

        J_corner_wrt_world_T_obj_point = GtsamMatrix(2, 3)
        J_corner_wrt_camera_pose = GtsamMatrix(2, 6)
        J_corner_wrt_camera_param = GtsamMatrix(2, 9)

        try:
            corner_predicted = camera.project(
                world_T_obj_pt,
                Dpose=J_corner_wrt_camera_pose,
                Dpoint=J_corner_wrt_world_T_obj_point,
                Dcal=J_corner_wrt_camera_param)
            errors.append(corner_predicted - tag_uv_point)
        except:
            errors.append(np.array([1, 1])*2*K.fx())

        J_uv_wrt_world_T_camera_pose_list.append(
            J_corner_wrt_camera_pose)
        J_uv_wrt_bundle_T_tag_pose_list.append(
            J_corner_wrt_world_T_obj_point@J_world_T_obj_pts_wrt_world_T_tag@J_world_T_tag_pose_wrt_bundle_T_tag_pose)
        J_uv_wrt_world_T_bundle_pose_list.append(
            J_corner_wrt_world_T_obj_point@J_world_T_obj_pts_wrt_world_T_tag@J_world_T_tag_pose_wrt_world_T_bundle_pose)

    if H is not None:
        H[0] = np.concatenate(J_uv_wrt_world_T_camera_pose_list)
        H[1] = np.concatenate(J_uv_wrt_bundle_T_tag_pose_list)
        H[2] = np.concatenate(J_uv_wrt_world_T_bundle_pose_list)
        # print(H[0].shape, H[1].shape, H[2].shape)
    return np.concatenate(errors)


def BundleTagFactor(corners: list[np.ndarray], k: gtsam.Cal3DS2, tag_size: float, world_T_camera_pose_key: int, bundle_T_tag_pose_key: int, world_T_bundle_pose_key: int, noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_camera_pose_key,
            bundle_T_tag_pose_key, world_T_bundle_pose_key]

    return gtsam.CustomFactor(keys=keys, noiseModel=noise_model, errorFunction=partial(bundle_projection_error, corners, k, tag_size))


def bundle_projection_error_i(pt_index: int, corners: list[np.ndarray], K: gtsam.Cal3DS2, tag_size: float, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    world_T_camera_pose_key = this.keys()[0]
    bundle_T_tag_pose_key = this.keys()[1]
    world_T_bundle_pose_key = this.keys()[2]
    world_T_camera_pose = v.atPose3(world_T_camera_pose_key)
    bundle_T_tag_pose = v.atPose3(bundle_T_tag_pose_key)
    world_T_bundle_pose = v.atPose3(world_T_bundle_pose_key)

    camera = gtsam.PinholePoseCal3DS2(pose=world_T_camera_pose, K=K)

    tag_T_obj_pts = [
        gtsam.Point3(-tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, tag_size / 2, 0),
        gtsam.Point3(-tag_size / 2, tag_size / 2, 0)]

    J_world_T_tag_pose_wrt_world_T_bundle_pose = GtsamMatrix(6, 6)
    J_world_T_tag_pose_wrt_bundle_T_tag_pose = GtsamMatrix(6, 6)
    world_T_tag_pose = world_T_bundle_pose.compose(
        bundle_T_tag_pose, J_world_T_tag_pose_wrt_world_T_bundle_pose, J_world_T_tag_pose_wrt_bundle_T_tag_pose)

    tag_T_obj_pt, tag_uv_point = tag_T_obj_pts[pt_index], corners[pt_index]
    J_world_T_obj_pts_wrt_world_T_tag = GtsamMatrix(3, 6)
    J_world_T_obj_pts_wrt_tag_T_points = GtsamMatrix(3, 3)

    world_T_obj_pt = world_T_tag_pose.transformFrom(
        tag_T_obj_pt, J_world_T_obj_pts_wrt_world_T_tag, J_world_T_obj_pts_wrt_tag_T_points)

    J_corner_wrt_world_T_obj_point = GtsamMatrix(2, 3)
    J_corner_wrt_camera_pose = GtsamMatrix(2, 6)
    J_corner_wrt_camera_param = GtsamMatrix(2, 9)

    try:
        corner_predicted = camera.project(
            world_T_obj_pt,
            Dpose=J_corner_wrt_camera_pose,
            Dpoint=J_corner_wrt_world_T_obj_point,
            Dcal=J_corner_wrt_camera_param)
        errors = corner_predicted - tag_uv_point

    except:
        errors = np.array([1, 1])*2*K.fx()

    if H is not None:
        H[0] = J_corner_wrt_camera_pose
        H[1] = J_corner_wrt_world_T_obj_point@J_world_T_obj_pts_wrt_world_T_tag@J_world_T_tag_pose_wrt_bundle_T_tag_pose
        H[2] = J_corner_wrt_world_T_obj_point@J_world_T_obj_pts_wrt_world_T_tag@J_world_T_tag_pose_wrt_world_T_bundle_pose
        # print(H[0].shape, H[1].shape, H[2].shape)
    return errors


def BundleTagFactors(corners: list[np.ndarray], k: gtsam.Cal3DS2, tag_size: float, world_T_camera_pose_key: int, bundle_T_tag_pose_key: int, world_T_bundle_pose_key: int, noise_model: gtsam.noiseModel.Gaussian):
    keys = [world_T_camera_pose_key,
            bundle_T_tag_pose_key, world_T_bundle_pose_key]
    factors = gtsam.NonlinearFactorGraph()
    for i in range(4):
        factors.push_back(
            gtsam.CustomFactor(keys=keys, noiseModel=noise_model, errorFunction=partial(
                bundle_projection_error_i, i, corners, k, tag_size))

        )
    return factors

import symforce  # noqa
symforce.set_epsilon_to_symbol()  # noqa


import symforce.symbolic as sf
import numpy as np
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce import typing as sfT

from symforce.opt import noise_models as nm

def identity_pose_residual(pose: sf.Pose3, epsilon: sf.Scalar):
    return sf.V6(pose.local_coordinates(sf.Pose3(), epsilon=epsilon))

def fixed_pose_residual(pose: sf.Pose3, fixed_pose: sf.Pose3, epsilon: sf.Scalar):
    error = pose.local_coordinates(fixed_pose, epsilon=epsilon)
    return sf.V6(error)


def prior_pose_residual(pose: sf.Pose3, prior_pose: sf.Pose3, epsilon: sf.Scalar, sigma: sf.Scalar):
    return pose.local_coordinates(prior_pose, epsilon=epsilon)


def prior_bewteen_pose_residual(poseA: sf.Pose3, poseB: sf.Pose3, prior_pose_ATB: sf.Pose3, epsilon: sf.Scalar):
    # robust_kernel=nm.BarronNoiseModel(2,0.1,x_epsilon=epsilon)
    error=sf.V6((poseA.inverse()*poseB).local_coordinates(prior_pose_ATB, epsilon=epsilon))
    # whitened_residual = (
    #     robust_kernel.whiten_norm(error, epsilon)
    # )   
    return error


def prior_bundle_pose_residual(world_T_camera: sf.Pose3, world_T_bundle: sf.Pose3, bundle_T_tag: sf.Pose3, prior_camera_T_tag: sf.Pose3, epsilon: sf.Scalar, sigma: sf.Scalar):
    return (world_T_camera.inverse()*world_T_bundle*bundle_T_tag).local_coordinates(prior_camera_T_tag, epsilon=epsilon)

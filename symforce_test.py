import symforce
symforce.set_epsilon_to_symbol()


import symforce.symbolic as sf
import numpy as np
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer



# def tag_residual(posed_camera: sf.PosedCamera, bundle_pose: sf.Pose3, tag_pose: sf.Pose3, tag_size: sf.Scalar, img_corners_points: sf.Matrix42, epsilon: sf.Scalar):
#     pass


def prior_pose_residual(pose: sf.Pose3, prior_pose: sf.Pose3, epsilon: sf.Scalar):
    return pose.local_coordinates(prior_pose, epsilon=epsilon)


factors = []
optimizer_key = []
initial_values = Values(
    poses=Values(),
    mearuments=Values(),
    epsilon=sf.numeric_epsilon,
)
ground_truth_pose_list = [
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 0, 0)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 1, 0)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 1, 2)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 1, 5)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(0, 4, 5)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 4, 5)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 4, 5)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 4, 0)),
    sf.Pose3(R=sf.Rot3.random(), t=sf.V3(1, 0, 0)),
    ]


gt_pose_list=Values()
for i, ground_truth_pose in enumerate(ground_truth_pose_list):
    pose_key = f"poses.p{i}"
    mea_key = f"mearuments.m{i}"

    initial_values[pose_key] = sf.Pose3()
    initial_values[mea_key] = ground_truth_pose
    gt_pose_list[pose_key] = ground_truth_pose
    # print(initial_values[pose_key])
    # print(ground_truth_pose)

    optimizer_key.append(pose_key)
    factors.append(Factor(
        residual=prior_pose_residual,
        keys=[pose_key, mea_key, "epsilon"],
    ))
    
    
optimizer = Optimizer(
    factors=factors,
    optimized_keys=optimizer_key,
    # So that we save more information about each iteration, to visualize later:
    debug_stats=True,
)

result = optimizer.optimize(initial_values)
assert result.status == Optimizer.Status.SUCCESS
# print(result.error())



# for key in result.optimized_values.keys_recursive():
#     if key in optimizer_key:
#         gt_pose:sf.Pose3 = gt_pose_list[key]
#         # init_pose=sf.Pose3(R=init_pose.rotation(),t=init_pose.position())
        
#         opt_pose:sf.Pose3 = result.optimized_values[key]
#         opt_pose=sf.Pose3.from_storage(opt_pose.to_storage())

#         # print(type(opt_pose))
        
#         # print(type(init_pose))
#         # print(type(init_result_pose))
        
#         # print(f" init:{init_pose} opt:{opt_pose} ")
        
#         print((gt_pose.inverse()*opt_pose).position().norm())
#         # print(key, result.optimized_values[key])


# # values_per_iter = [optimizer.load_iteration_values(
# #     stats.values) for stats in result.iterations]
# # for values in values_per_iter:
# #     sum_error = 0
# #     for key in values.keys_recursive():
# #         if key in optimizer_key:
# #             init_pose:sf.Pose3 = initial_values[key]
# #             opt_pose:sf.Pose3 = values[key]
# #             print(type(init_pose))
# #             print(type(opt_pose))
# #             sum_error += (init_pose.inverse()*opt_pose).position().norm()
# #     print(sum_error)

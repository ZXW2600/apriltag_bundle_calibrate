
import apriltag_calibrate.custom_factor.bundle_custom_factor as bundle_custom_factor
import gtsam
import numpy as np
from apriltag_calibrate.custom_factor.bundle_custom_factor import BundlePoseFactor, BundleTagFactor


def random_rot():
    return gtsam.Rot3.AxisAngle(np.random.randn(3), np.random.rand()*360)


def random_pose():
    return gtsam.Pose3(gtsam.Rot3.AxisAngle(np.random.randn(3), np.random.rand()*360), gtsam.Point3(10*np.random.randn(3)))


tag_pose_list = [
    random_pose() for i in range(100)
]

world_T_camera_pose = gtsam.Pose3(random_rot(), np.array([0, 0, 0]))

camera_T_tag_pose_list = [
    world_T_camera_pose.between(tag_pose)
    for tag_pose in tag_pose_list]


# add noise  to tag pose
tag_pose_noise_list = []
for tag_pose in tag_pose_list:
    tag_pose_noise_list.append(tag_pose.compose(gtsam.Pose3(
        gtsam.Rot3.Rodrigues(np.random.randn(3)*0.05), np.random.randn(3)*0.1)))
    print(
        f"noise {np.linalg.norm(tag_pose.between(tag_pose_noise_list[-1]).translation())}")


def test_tag_bundle_factor():
    tag_size = 0.1
    K = gtsam.Cal3DS2(100, 100, 0, 0, 0, 0, 0, 0, 0)
    tag_T_obj_pts = [
        gtsam.Point3(-tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, -tag_size / 2, 0),
        gtsam.Point3(tag_size / 2, tag_size / 2, 0),
        gtsam.Point3(-tag_size / 2, tag_size / 2, 0)]

    camera_pose = gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 0]))
    camera = gtsam.PinholePoseCal3DS2(pose=camera_pose, K=K)

    world_T_tag_points_list = []
    uv_list = []
    for tag_pose in tag_pose_list:
        world_T_tag_points_list.append(
            [tag_pose.transformFrom(pt) for pt in tag_T_obj_pts])
        try:
            uv_list.append([camera.project(pt)
                        for pt in world_T_tag_points_list[-1]])
        except:
            pass
        
    graph = gtsam.NonlinearFactorGraph()
    init_values = gtsam.Values()

    camera_key = gtsam.symbol('c', 0)
    init_values.insert(camera_key, gtsam.Pose3())

    bundle_key = gtsam.symbol('b', 0)
    init_values.insert(bundle_key, gtsam.Pose3())

    for i, tag_uv in enumerate(uv_list):
        tag_key = gtsam.symbol('t', i)
        init_values.insert(tag_key, tag_pose_noise_list[i])
        graph.push_back(BundleTagFactor(tag_uv, K, tag_size, camera_key,
                        tag_key, bundle_key, gtsam.noiseModel.Isotropic.Sigma(8, 0.1)))
        graph.add
        if i == 0:
            graph.push_back(gtsam.PriorFactorPose3(
                tag_key, gtsam.Pose3(), gtsam.noiseModel.Constrained.All(6)))

    # solver
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
    result = optimizer.optimize()


def test_bundle_pose_factor():
    bundle_tag_pose_list = []
    first_tag_pose = tag_pose_list[0]

    for tag_pose in tag_pose_list:
        bundle_tag_pose_list.append(first_tag_pose.between(tag_pose))

    # print(tag_pose_list)

    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()
    init_values = gtsam.Values()

    # add a bundle
    world_T_bundle_pose_key = gtsam.symbol('b', 0)
    init_values.insert_pose3(world_T_bundle_pose_key, gtsam.Pose3())

    # add tags to the bundle
    for i, tag_pose in enumerate(tag_pose_list):
        bundle_T_tag_pose_key = gtsam.symbol('t', i)
        world_T_tag_pose_key = gtsam.symbol('w', i)

        if i == 0:
            fix_noise = gtsam.noiseModel.Constrained.All(6)
            graph.push_back(gtsam.PriorFactorPose3(
                bundle_T_tag_pose_key, gtsam.Pose3(), fix_noise))

        bundle_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        graph.push_back(BundlePoseFactor(world_T_bundle_pose_key,
                        bundle_T_tag_pose_key, world_T_tag_pose_key, bundle_noise))
        init_values.insert(bundle_T_tag_pose_key, tag_pose)
        init_values.insert(world_T_tag_pose_key, tag_pose)

        graph.push_back(gtsam.PriorFactorPose3(
            world_T_tag_pose_key, tag_pose, bundle_noise))

    # solver
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
    result = optimizer.optimize()

    print("world tag pose")
    for i, world_T_tag_pose in enumerate(tag_pose_list):
        print(np.linalg.norm(world_T_tag_pose.between(
            result.atPose3(gtsam.symbol('w', i))).translation()))

    print("bundle tag pose")
    for i, bundle_T_tag_pose in enumerate(bundle_tag_pose_list):
        print(np.linalg.norm(bundle_T_tag_pose.between(
            result.atPose3(gtsam.symbol('t', i))).translation()))


def test_pnp_factor():

    graph = gtsam.NonlinearFactorGraph()
    init_values = gtsam.Values()

    bundle_key = gtsam.symbol('b', 0)

    camera_key = gtsam.symbol('c', 0)
    init_values.insert(camera_key, gtsam.Pose3())

    tag_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1),
        # Adjust these values as needed
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
    )

    for i, tag_pose in enumerate(camera_T_tag_pose_list):
        tag_key = gtsam.symbol('t', i)
        graph.push_back(
            bundle_custom_factor.BundleCameraPnPFactor(
                bundle_key, tag_key, camera_key, tag_pose, tag_noise
            )
        )
        init_values.insert(tag_key, gtsam.Pose3())
        if i == 0:
            graph.push_back(gtsam.PriorFactorPose3(
                tag_key, gtsam.Pose3(), gtsam.noiseModel.Constrained.All(6)))
            init_values.insert(bundle_key, tag_pose)
            # graph.push_back(
            #     gtsam.PriorFactorPose3(
            #         bundle_key, gtsam.Pose3(), gtsam.noiseModel.Constrained.All(6))
            # )

    # solver
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
    result = optimizer.optimize()

    print("world tag pose")
    # world_T_bundle = result.atPose3(bundle_key)
    # for i, bundle_T_tag_pose in enumerate(tag_pose_list):
    #     world_T_tag_pose = world_T_bundle.compose(bundle_T_tag_pose)
    #     print(np.linalg.norm(world_T_tag_pose.between(
    #         result.atPose3(gtsam.symbol('t', i))).translation()))


def test_pnp_bunde_gtsam():
    graph = gtsam.NonlinearFactorGraph()
    init_values = gtsam.Values()

    bundle_key = gtsam.symbol('b', 0)
    init_values.insert(bundle_key,  gtsam.Pose3())

    camera_key = gtsam.symbol('c', 0)
    init_values.insert(camera_key, gtsam.Pose3())

    tag_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1),
        # Adjust these values as needed
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
    )

    for i, tag_pose in enumerate(camera_T_tag_pose_list):
        tag_key = gtsam.symbol('t', i)
        tag_bundle_ikey = gtsam.symbol('w', i)
        graph.push_back(
            bundle_custom_factor.BundlePoseFactor(
                bundle_key, tag_key, tag_bundle_ikey, gtsam.noiseModel.Isotropic.Sigma(
                    6, 0.1)
            )
        )
        init_values.insert(
            tag_key, gtsam.Pose3()
        )
        init_values.insert(
            tag_bundle_ikey, gtsam.Pose3()
        )

        graph.push_back(
            gtsam.BetweenFactorPose3(
                camera_key, tag_bundle_ikey, tag_pose, tag_noise
            )
        )
        
        if i ==0:
            graph.push_back(
                gtsam.PriorFactorPose3(
                    tag_key, gtsam.Pose3(), gtsam.noiseModel.Constrained.All(6)
                )
            )
            graph.push_back(
                gtsam.PriorFactorPose3(
                    tag_bundle_ikey, gtsam.Pose3(), gtsam.noiseModel.Constrained.All(6)
                )
            )
    
    # solver
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
    result = optimizer.optimize()

    print("world tag pose")
    world_T_bundle = result.atPose3(bundle_key)
        

test_tag_bundle_factor()

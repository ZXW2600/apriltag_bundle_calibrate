import gtsam
import gtsam_unstable
import numpy as np

graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
]
pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
pose_symbol = gtsam.symbol('x', 0)
initial_estimate.insert(pose_symbol, pose)
graph.push_back(gtsam.PriorFactorPose3(pose_symbol, pose,
                gtsam.noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])))
for i in range(len(points)):
    p = points[i]
    symbol = gtsam.symbol('p', i)
    initial_estimate.insert(symbol, gtsam.Point3(p[0]+0.1, p[1]+0.2, p[2]+0.3))
    graph.push_back(
        gtsam_unstable.Pose3ToPoint3Factor(
            pose_symbol, symbol, np.array(
                [p[0], p[1], p[2]]), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        )
    )


# solve
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY")
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()

# print result
for i in range(len(points)):
    print(f"opt: {result.atPoint3(gtsam.symbol('p', i))} gt : {points[i]}")
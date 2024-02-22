import numpy as np


def getTagPoints(tag_pose, tag_size):
    tag_size_2 = tag_size / 2.0
    vertex_std = np.array([[-tag_size_2, -tag_size_2, 0, 1],
                           [tag_size_2, -tag_size_2, 0, 1],
                           [tag_size_2, tag_size_2, 0, 1],
                           [-tag_size_2, tag_size_2, 0, 1]])
    vertex_world = vertex_std@tag_pose.T
    return vertex_world[:, :-1]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def draw_camera(ax, pose: np.ndarray, focal_len_scaled=0.10, aspect_ratio=0.3, color: str = 'b'):
    vertex_std = np.array([[0, 0, 0, 1],
                           [focal_len_scaled * aspect_ratio, -focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [focal_len_scaled * aspect_ratio, focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
    vertex_world = pose@vertex_std.T
    vertex_world = vertex_world.T
    meshes = [[vertex_world[0, :-1], vertex_world[1][:-1], vertex_world[2, :-1]],
              [vertex_world[0, :-1], vertex_world[2, :-1],
                  vertex_world[3, :-1]],
              [vertex_world[0, :-1], vertex_world[3, :-1],
                  vertex_world[4, :-1]],
              [vertex_world[0, :-1], vertex_world[4, :-1],
                  vertex_world[1, :-1]],
              [vertex_world[1, :-1], vertex_world[2, :-1], vertex_world[3, :-1], vertex_world[4, :-1]]]

    # vertex_transformed = vertex_std @ pose
    # meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
    #           [vertex_transformed[0, :-1], vertex_transformed[2, :-1],
    #               vertex_transformed[3, :-1]],
    #           [vertex_transformed[0, :-1], vertex_transformed[3, :-1],
    #               vertex_transformed[4, :-1]],
    #           [vertex_transformed[0, :-1], vertex_transformed[4, :-1],
    #               vertex_transformed[1, :-1]],
    #           [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
    ax.add_collection3d(
        Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))


def draw_3dpoints(ax, points: np.ndarray, size: float = 0.01, line_width=2, color: str = 'g'):
    lines = []
    for d in [(0, 0, 0.5*size), (0, 0.5*size, 0), (0.5*size, 0, 0)]:
        lines.append([points-d, points+d])
    ax.add_collection3d(
        Line3DCollection(lines, facecolors=color, linewidths=line_width, edgecolors=color, alpha=0.35))


def draw_axes(ax, pose, size: float = 0.1, line_width=2):
    axes_T_end_points = [
        np.array([size, 0, 0, 1]),
        np.array([-size, 0, 0, 1]),
        np.array([0, size, 0, 1]),
        np.array([0, -size, 0, 1]),
        np.array([0, 0, size, 1]),
        np.array([0, 0, -size, 1]),
    ]
    world_T_end_points = (pose@np.array(axes_T_end_points).T).T
    # print(world_T_end_points)
    color = ["r", "g", "b"]
    for i in range(0, 3, 1):
        start = world_T_end_points[i*2, :3]
        end = world_T_end_points[i*2+1, :3]
        # print(start, end)
        ax.add_collection3d(
            Line3DCollection([[start, end]], facecolors=color, linewidths=line_width, edgecolors=color[i], alpha=0.35))

tag_color_dict = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (0, 0, 1),
    3: (1, 1, 0),
    4: (1, 0, 1),
    5: (0, 1, 1)
}


def get_tag_color(tag_id):
    if tag_id not in tag_color_dict:
        tag_color_dict[tag_id] = np.random.rand(3,)
    return tag_color_dict[tag_id]


def draw_tag(ax, pose, tag_size, tag_id, color='r'):
    tag_size_2 = tag_size / 2.0
    vertex_std = np.array([[-tag_size_2, -tag_size_2, 0, 1],
                           [tag_size_2, -tag_size_2, 0, 1],
                           [tag_size_2, tag_size_2, 0, 1],
                           [-tag_size_2, tag_size_2, 0, 1]])

    vertex_world = pose@vertex_std.T
    # print(vertex_std)

    vertex_world = vertex_world.T
    # print(vertex_world)
    meshes = np.array([[vertex_world[0, :-1], vertex_world[1, :-1],
                        vertex_world[2, :-1], vertex_world[3, :-1]]])
    # print(meshes.shape)

    ax.add_collection3d(
        Poly3DCollection(meshes, facecolors=get_tag_color(tag_id), linewidths=0.3, edgecolors=color, alpha=0.35))
    return vertex_world
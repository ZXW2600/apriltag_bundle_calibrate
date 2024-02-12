import yaml
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_camera(ax, pose: np.ndarray, focal_len_scaled=0.10, aspect_ratio=0.3, color: str = 'b'):
    vertex_std = np.array([[0, 0, 0, 1],
                           [focal_len_scaled * aspect_ratio, -focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [focal_len_scaled * aspect_ratio, focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
    vertex_world=pose@vertex_std.T
    vertex_world=vertex_world.T
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


tag_color_dict = {}


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
    
    vertex_world=pose@vertex_std.T
    print(vertex_std)
    
    vertex_world=vertex_world.T
    print(vertex_world)
    meshes = np.array([[vertex_world[0, :-1], vertex_world[1, :-1],
                        vertex_world[2, :-1], vertex_world[3, :-1]]])
    print(meshes.shape)

    ax.add_collection3d(
        Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))


figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

file_path = "/home/zxw2600/Workspace_Disk/inertia_toolkit_ws/apriltag_calib_ws/bundle_calib_raw.yaml"
with open(file_path, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

    # get camera data
    index_cam = 0
    while True:
        if index_cam in data:
            figure = plt.figure()
            ax = figure.add_subplot(111, projection='3d')
            draw_camera(ax, np.diag([1, 1, 1, 1]), color='b')
            tag_pose_dict = data[index_cam]
            for tag_id, tag_pose in tag_pose_dict.items():
                pose=np.array(tag_pose)
                print(pose)
                draw_tag(ax, pose,  0.0285,tag_id)
            index_cam += 1
            plt.show()
            
        else:
            break


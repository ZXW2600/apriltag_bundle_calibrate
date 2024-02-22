import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
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
    print(world_T_end_points)
    color = ["r", "g", "b"]
    for i in range(0, 3, 1):
        start = world_T_end_points[i*2, :3]
        end = world_T_end_points[i*2+1, :3]
        print(start, end)
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

vis_camera = True
vis_master_tag = True
vis_aid_tag = False
vis_points = True
test_plane = False
test_cube = True

if test_plane:
    vis_points = True


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


figure = plt.figure()
ax = figure.add_subplot(121, projection='3d')
ax_tag = figure.add_subplot(122, projection='3d')
# ax.set_box_aspect([1, 1, 1])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the bundle calibrate file ")

ap.add_argument("-s", "--size", required=False,
                default=0.015, help="size of the tag")
tag_size = float(ap.parse_args().size)
file_path = ap.parse_args().input
points = []
tags = []
z_s = []
with open(file_path, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

    for tag in data:
        chr = tag[0]
        index = int(tag[1:])
        pose = np.array(data[tag])
        if chr == 't' and vis_master_tag:
            points = draw_tag(ax_tag, pose, tag_size, index)
            tags.append((pose, points))
        elif chr == 'x' and vis_camera:
            draw_camera(ax, pose, 0.05, 0.2)
        elif chr == 'a' and vis_aid_tag:
            draw_tag(ax_tag, pose, tag_size, index)
        elif chr == 'p' and vis_points:
            pose.reshape(3)
            draw_3dpoints(ax, pose)
            points.append(pose)
        elif chr == "b":
            draw_axes(ax, pose, 0.1)

if test_plane:
    # fit plane
    points = np.array(points)

    # Calculate the mean of the points
    centroid = np.mean(points, axis=0)

    # Subtract the mean from the points
    points_centered = points - centroid

    # Perform singular value decomposition
    u, s, vh = np.linalg.svd(points_centered)

    # The normal vector of the plane is the last column in vh
    normal = vh[-1]

    # The coefficients of the plane equation are the elements of the normal vector
    # The constant term is given by the dot product of the centroid and the normal vector
    a, b, c = normal
    d = -centroid.dot(normal)

    # Calculate the distances from the points to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] +
                       c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

    # Calculate the fit error as the sum of the squared distances
    fit_error = np.sum(distances**2)

    print(f"The plane fit error is {fit_error}")


if test_cube:
    figure_cube = plt.figure()
    ax_cube = figure_cube.add_subplot(111, projection='3d')

    # Split tags into six planes according to pose and points
    pose_group = [[], [], [], [], [], []]
    pose_ref = [
        np.array([0, 0, 1, 1],),
        np.array([0, 1, 0, 1],),
        np.array([1, 0, 0, 1]),
    ]
    for pose, points in tags:
        min_d = 100
        min_dd = 100
        min_index = 0

        R = pose[:3, :3]

        for i in range(3):
            ref_vec = pose_ref[i][:-1]
            pose_vec: np.ndarray = R@np.array([0, 0, 1]).T
            dd = np.dot(ref_vec, pose_vec)
            d = np.abs(1-np.abs(dd))
            if d < 0:
                pose_vec = -pose_vec
            # print(f"{i} {d}")
            if d < min_d:
                min_d = d
                min_dd = dd
                min_index = i
        if min_dd < 0:
            pose_group[min_index].append(
                (pose, points, pose_ref[min_index][:-1]))
        else:
            pose_group[min_index +
                       3].append((pose, points, pose_ref[min_index][:-1]))
    group_vec = []
    for group_id, group in enumerate(pose_group):
        avr_vec = []
        print("_group start_______")
        for pose, points, ref_vec in group:
            R = pose[:3, :3]
            pose_vec: np.ndarray = R@np.array([0, 0, 1]).T
            d = np.dot(ref_vec, pose_vec)
            if d < 0:
                pose_vec = -pose_vec

            print(f"vec {pose_vec.round(2)} {d:2f}")
            draw_tag(ax_cube, pose, tag_size, group_id)
            avr_vec.append(pose_vec)
        group_vec.append(np.mean(avr_vec, axis=0))
    near_angles = []
    opsite_angles = []
    for i in range(len(group_vec)):
        for j in range(i+1, len(group_vec)):
            # print(f"{i} {j} {np.dot(group_vec[i], group_vec[j])}")
            rad = np.arccos(np.dot(group_vec[i], group_vec[j]))
            if j-i == 3:
                opsite_angles.append(rad*180.0/np.pi)
            else:
                near_angles.append(rad*180.0/np.pi)
    print(f"near angles {near_angles}")
    print(f"opsite angles {opsite_angles}")
    oppsite_error = np.abs(np.array(opsite_angles)-0.0)
    near_error = np.abs(np.array(near_angles)-90.0)
    print("max error ", max(oppsite_error.max(), near_error.max()))
    print(f"avr error", np.mean(np.concatenate((oppsite_error, near_error))))
    
    
    
    
# Set the axes to be tight to maximize the plot area
plt.axis('tight')

# Remove padding and margin
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

plt.show()

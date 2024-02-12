import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection


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


def draw_3dpoints(ax, points: np.ndarray,size:float=0.01,line_width=2 ,color: str = 'g'):
    lines=[]
    for d in [(0,0,0.5*size),(0,0.5*size,0),(0.5*size,0,0)]:
        lines.append([points-d,points+d])
    ax.add_collection3d(
        Line3DCollection(lines, facecolors=color, linewidths=line_width, edgecolors=color, alpha=0.35))


tag_color_dict = {}

vis_camera = True
vis_master_tag = True
vis_aid_tag = True
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


figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')
# ax.set_box_aspect([1, 1, 1])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the bundle calibrate file ")


file_path = ap.parse_args().input
points = []
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
            draw_tag(ax, pose, 0.0194, index)
        elif chr == 'x' and vis_camera:
            draw_camera(ax, pose, 0.05, 0.2)
        elif chr == 'a' and vis_aid_tag:
            draw_tag(ax, pose, 0.0155, index)
        elif chr == 'p' and vis_points:
            pose.reshape(3)
            draw_3dpoints(ax, pose)
            points.append(pose)
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
distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

# Calculate the fit error as the sum of the squared distances
fit_error = np.sum(distances**2)

print(f"The plane fit error is {fit_error}")

plt.show()

from moms_apriltag import ApriltagBoard
import cv2
import numpy as np
import argparse
import yaml
import os
import apriltag
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# get image path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="folder path to the input image",
                default="/home/zxw2600/Workspace_Disk/inertia_toolkit_ws/apriltag_calib_ws/img")
ap.add_argument("-o", "--output", required=False,
                help="output file name", default="camera_calib.yaml")
folder_path = ap.parse_args().image
yaml_file = ap.parse_args().output

# read all the image to a list
image_files = os.listdir(folder_path)
for file in image_files:
    if not file.endswith(".jpg") and not file.endswith(".png"):
        image_files.remove(file)

# images = []
def load_img(file):
    image_path = os.path.join(folder_path, file)
    image = cv2.imread(image_path)
    if image is not None:
        return image
        
print("reading images...")
with ProcessPoolExecutor() as executor:
    images=list(tqdm(executor.map(load_img, image_files), total=len(image_files)))



# create apriltag board

print("reading apriltag config...")

class ApriltagBoard:
    def __init__(self):
        self.objPoints = {}
        
        pass
    def read_yaml(self, file):
        with open(file, 'r') as stream:
            data = yaml.safe_load(stream)
            for tag_id, tag_data in data.items():
                center = np.array(tag_data['center'], np.float32)
                corners = [np.array(c, np.float32)
                           for c in tag_data['corners']]
                self.objPoints[int(tag_id)] = corners


options = apriltag.DetectorOptions(families='tag25h9',
                                   border=1,
                                   nthreads=4,
                                   quad_decimate=4,
                                   quad_blur=0.0,
                                   refine_edges=True,
                                   refine_decode=False,
                                   refine_pose=False,
                                   debug=False,
                                   quad_contours=True)

# detect apriltag in the image
at_detector = apriltag.Detector(options)

board = ApriltagBoard()
board.read_yaml(
    "/home/zxw2600/Workspace_Disk/inertia_toolkit_ws/apriltag_calib_ws/apriltag_board.yaml")

board_points_show = {}

offset = np.array([1, 1], np.float32)
for index, corner in board.objPoints.items():
    board_points_show[index] = [p[:2]*20000 +offset for p in corner]

img_size = images[0].shape[:2]

obj_points = []
img_points = []

vis = False
print("process img...")
for img in tqdm(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)
    img_obj_points=[]
    img_img_points=[]
    
    for result in results:
        index = result.tag_id
        for i in range(4):
            img_img_points.append(result.corners[i].astype(np.float32))
            img_obj_points.append(board.objPoints[index][i].astype(np.float32))

    obj_points.append(np.asarray(img_obj_points))
    img_points.append(np.asarray(img_img_points))
    
    if vis:
        img_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        board_show = np.zeros_like(img_show)
        cv2.namedWindow('img', cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow('board', cv2.WINDOW_GUI_NORMAL)

        for result in results:
            index = result.tag_id
            corner = result.corners
            center = result.center

            board_points = board_points_show[index]
            board_center = np.mean(board_points, axis=0)

            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            cv2.putText(img_show, str(index), tuple(center.astype(
                int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 50), 3)
            cv2.putText(board_show, str(index), tuple(board_center.astype(int))[
                        :2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 50), 3)

            for i in range(4):
                cv2.circle(img_show, tuple(
                    corner[i].astype(int)), 3, color[i], -1)
                cv2.circle(board_show, tuple(
                    board_points[i].astype(int))[:2], 5, color[i], -1)
                cv2.putText(img_show, str(i), tuple(corner[i].astype(
                    int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(board_show, str(i), tuple(board_points[i].astype(int))[
                            :2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('img', img_show)
            cv2.imshow('board', board_show)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
cv2.destroyAllWindows()

print("calculate parameters...")
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None)

print("Camera Matrix : \n", cameraMatrix)
print("Camera Distort coeff : \n", distCoeffs)
cx = float(cameraMatrix[0, 2])
cy = float(cameraMatrix[1, 2])
fx = float(cameraMatrix[0, 0])
fy = float(cameraMatrix[1, 1])

camera_param = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "distCoeffs": distCoeffs.tolist(), "cameraMatrix": cameraMatrix.tolist()}
# write resule to file
with open(yaml_file, 'w') as file:
    documents = yaml.dump(camera_param, file)
    print(f"Camera calibration result is saved to {yaml_file}")


print("test undistort img...")
cv2.namedWindow('undistorted', cv2.WINDOW_GUI_NORMAL)
# show undistorted img
for img in images:
    dst = cv2.undistort(img, cameraMatrix, distCoeffs)
    cv2.imshow('undistorted', dst)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

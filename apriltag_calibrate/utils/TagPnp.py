import apriltag
import cv2
import numpy as np
from apriltag_calibrate.configparase.TagBundle import TagBundle
from apriltag_calibrate.configparase.Camera import Camera


class TagPnP:
    def __init__(self) -> None:
        self.obj_points = []
        self.img_points = []

    def add_tag(self, tags: list[apriltag.Detection], tag_bundle: TagBundle):
        for tag in tags:
            if tag.tag_id in tag_bundle.tag_keys:
                for i in range(4):
                    self.obj_points.append(
                        tag_bundle.tag_points[tag.tag_id][i])
                    self.img_points.append(tag.corners[i])

    def solve(self, camera: Camera):
        ret, rvecs, tvecs = cv2.solvePnP(np.array(self.obj_points), np.array(
            self.img_points), camera.cameraMatrix, camera.distCoeffs)
        return ret, rvecs, tvecs

import numpy as np
import yaml

class Camera:
    def __init__(self, camera_param_file) -> None:
        # read camera parameters
        with open(camera_param_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            self.cx = data["cx"]
            self.cy = data["cy"]
            self.fx = data["fx"]
            self.fy = data["fy"]
            self.distCoeffs = np.array(data["distCoeffs"]).flatten()
            if self.distCoeffs.shape[0] > 4:
                self.k1 = self.distCoeffs[0]
                self.k2 = self.distCoeffs[1]
                self.p1 = self.distCoeffs[2]
                self.p2 = self.distCoeffs[3]

            self.cameraMatrix = np.array(
                [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3)
        print(f"Camera parameters are loaded from {camera_param_file}"
              f"\nfx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, distCoeffs={self.distCoeffs}")


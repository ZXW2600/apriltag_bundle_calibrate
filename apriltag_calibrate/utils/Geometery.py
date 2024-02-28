import numpy as np
from cv2 import Rodrigues

def Rtvec2HomogeousT(rvecs,tvecs):
    pose=np.eye(4)
    pose[:3,:3]=Rodrigues(rvecs)[0]
    pose[:3,3]=tvecs.T
    return pose
from apriltag_calibrate.utils.Tag import getTagPoints
import numpy as np


R=np.eye(3)
t=np.array([2,2,2])
RT=np.eye(4)
RT[:3,:3]=R
RT[:3,3]=t

getTagPoints(RT, 1.0)
print(getTagPoints(RT, 1.0))
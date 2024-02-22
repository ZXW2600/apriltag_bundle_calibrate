import yaml
from apriltag_calibrate.utils.Tag import getTagPoints
from apriltag_calibrate.utils  import  KeyType
import numpy as np

class TagBundle:
    def __init__(self) -> None:
        pass
    
    def load(self, tag_bundle_file):
        with open(tag_bundle_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        bundle_pose = data["bundle_pose"]
        camera_pose=data["camera_pose"]
        extra_data=data["extra_data"]
        tag_pose=data["tag_pose"]
        
        self.tag_family=extra_data["tag_family"]
        self.tag_size=extra_data["tag_size"]
        
        self.bundle_pose={}
        self.camera_pose={}
        self.tag_pose={}
        self.tag_points={}
        for tag, data in tag_pose.items():
            chr = tag[0]
            index = int(tag[1:])
            pose = np.array(data)
            self.tag_pose[tag]=pose
            pts_np:np.ndarray=getTagPoints(pose, self.tag_size)
            self.tag_points[index]=[
                pts_np[i,] for i in range(4)
            ]
            
        for bundle, data in bundle_pose.items():
            chr = bundle[0]
            index = int(bundle[1:])
            pose = np.array(data)
            self.bundle_pose[index]=pose
        
        for camera, data in camera_pose.items():
            chr = camera[0]
            index = int(camera[1:])
            pose = np.array(data)
            self.camera_pose[index]=pose
            
        self.tag_keys = self.tag_points.keys()
        
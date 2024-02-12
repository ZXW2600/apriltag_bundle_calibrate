
import yaml
import numpy as np

class ApriltagBoard:
    def __init__(self):
        self.objPoints = {}

        pass

    def read_yaml(self, file):

        with open(file, 'r') as stream:
            data = yaml.safe_load(stream)
            self.cols = data['col']
            self.rows = data['row']
            self.tag_family = data['tag_family']
            self.tag_obj_points = data['tag_obj_points']
            for tag_id, tag_data in self.tag_obj_points.items():
                center = np.array(tag_data['center'], np.float32)
                corners = [np.array(c, np.float32)
                           for c in tag_data['corners']]
                self.objPoints[int(tag_id)] = corners
import os
from .target_dataset import TargetDataset


class Mapillary(TargetDataset):

    task = 'segmentation'
    ds_type = 'supervised'
    labels2train = {2: 1, 3: 4, 6: 3, 7: 1, 9: 1, 11: 1, 13: 0, 14: 0, 15: 1, 17: 2, 19: 11, 20: 12, 21: 12, 22: 12,
                    24: 0, 26: 9, 27: 10, 29: 9, 30: 8, 43: 0, 44: 5, 45: 5, 46: 5, 47: 5, 48: 6, 50: 7, 52: 18,
                    54: 15, 55: 13, 57: 17, 58: 16, 61: 14}
    images_dir = os.path.join('mapillary', 'data')
    target_dir = os.path.join('mapillary', 'data')

    def __init__(self, paths, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, hp_filtered=False):
        super().__init__(paths, root, transform=transform, test_transform=test_transform, mean=mean, std=std, cv2=cv2,
                         hp_filtered=hp_filtered)


"""
mapillary                       -->     cityscapes 19
----------------------------------------------------------------
Curb 2                          -->     sidewalk 1
Fence 3                         -->     fence 4
Wall 6                          -->     wall 3
bike lane 7                     -->     sidewalk 1
curb cut 9                      -->     sidewalk 1
Pedestrian Area 11              -->     sidewalk 1
road 13                         -->     road 0
Service Lane 14                 -->     road 0
Sidewalk 15                     -->     sidewalk 1
Building 17                     -->     building 2
Person 19                       -->     person 11
Bicyclist 20                    -->     rider 12
Motorcyclist 21                 -->     rider 12
Other Rider 22                  -->     rider 12
Lane Marking - General 24       -->     road 0
Sand 26                         -->     terrain 9
Sky 27                          -->     sky 10
Terrain 29                      -->     terrain 9
Vegetation 30                   -->     vegetation 8
Pothole 43                      -->     road 0
Street Light 44                 -->     pole 5
pole 45                         -->     pole 5
Traffic Sign Frame 46           -->     pole 5
Utility Pole 47                 -->     pole 5
Traffic Light 48                -->     traffic light 6
Traffic Sign (Front) 50         -->     traffic sign 7
Bicycle 52                      -->     bicycle 18
Bus 54                          -->     bus 15
Car 55                          -->     car 13
Motorcycle 57                   -->     motorcycle 17
On Rails 58                     -->     train 16
Truck 61                        -->     truck 14
"""

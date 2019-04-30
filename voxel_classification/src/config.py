#!/usr/bin/env python

"""VoxelNet config system.
"""
# from easydict import EasyDict as edict

# __C = edict()
# Consumers can get config by:
#    import config as cfg
# cfg = __C

class Cfg():

    def __init__(self):
        # for dataset dir
        self.ROOT_DIR = '/home/leo/phd/particles_detection_fsr/voxel_classification'
        self.LOG_DIR = '/home/leo/phd/particles_detection_fsr/voxel_classification/logs'
        self.RAW_DATA_DIR = '/home/leo/phd/smoke_detection/src/smoke_detection/lidar_classifier/data/eth'
        self.DATASETS_DIR = '/home/leo/phd/particles_detection_fsr/voxel_classification/data'


        # Size in [m]
        self.Z_SIZE = 3
        self.Y_SIZE = 20
        self.X_SIZE = 20
        #  When changing MAP_TO_VELO, update transforms.launch too for correct Rviz display
        self.MAP_TO_VELO_X = 10
        self.MAP_TO_VELO_Y = 10
        self.MAP_TO_VELO_Z = 2
        # Size in [m/voxel]
        self.VOXEL_X_SIZE = 0.2
        self.VOXEL_Y_SIZE = 0.2
        self.VOXEL_Z_SIZE = 0.2
        # Maximum number of lidar points per voxels
        self.VOXEL_POINT_COUNT = 35
        #
        self.VFE1_OUT = 32
        self.VFE2_OUT = 128

cfg = Cfg()
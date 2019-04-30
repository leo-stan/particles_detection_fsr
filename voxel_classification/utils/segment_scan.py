import sys
sys.path.insert(0, '../src')
from config import cfg
import numpy as np


def segment_scan(pcl, output, output_proba_mean=None, output_proba_std=None):

    # Translation from map to robot_footprint
    husky_footprint_coord = np.array([cfg.MAP_TO_VELO_X, cfg.MAP_TO_VELO_Y, cfg.MAP_TO_VELO_Z],
                                     dtype=np.float32)

    # Lidar points in map coordinate
    shifted_coord = pcl[:, :3] + husky_footprint_coord

    voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype=np.float32)
    grid_size = np.array([cfg.X_SIZE / cfg.VOXEL_X_SIZE, cfg.Y_SIZE / cfg.VOXEL_Y_SIZE,
                          cfg.Z_SIZE / cfg.VOXEL_Z_SIZE], dtype=np.int64)

    voxel_index = np.floor(shifted_coord / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    voxel_index = voxel_index[bound_box]

    # coordinate buffer
    coordinate_buffer = np.unique(voxel_index, axis=0)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(len(coordinate_buffer)):
        index_buffer[tuple(coordinate_buffer[i])] = i

    raw_pcl_label = np.zeros((len(voxel_index), 1))
    raw_pcl_proba_mean = np.zeros((len(voxel_index), 1))
    raw_pcl_proba_std = np.zeros((len(voxel_index), 1))
    for i in range(len(voxel_index)):
        raw_pcl_label[i] = output[index_buffer[tuple(voxel_index[i, :])]]
        if output_proba_mean:
            raw_pcl_proba_mean[i] = output_proba_mean[index_buffer[tuple(voxel_index[i, :])]]
        if output_proba_std:
            raw_pcl_proba_std[i] = output_proba_std[index_buffer[tuple(voxel_index[i, :])]]

    pred_raw_lidar_pcl = np.concatenate((pcl[bound_box,:], raw_pcl_label, raw_pcl_proba_mean, raw_pcl_proba_std), axis=1)

    return pred_raw_lidar_pcl
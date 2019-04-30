import sys
sys.path.insert(0, '../src')

import numpy as np
from config import cfg

def extract_scan(raw_lidar_pcl, features):
    """
    Puts raw lidar points into voxels and prepare data for training
    :param raw_lidar_pcl: numpy array with raw lidar points
    :return:
    """

    # if shuffle:
    #     np.random.shuffle(raw_lidar_pcl)

    # Process raw scan
    # Apply -90deg rotation on z axis to go from robot to map
    # raw_lidar_pcl[:, [0, 1]] = raw_lidar_pcl[:, [1, 0]]  # Swap x, y axis
    # raw_lidar_pcl[:, :3] = raw_lidar_pcl[:, :3] * np.array([-1, 1, 1], dtype=np.int8)

    # removes points at [0,0,0]
    # if np.any(np.sum(raw_lidar_pcl[:, :3], axis=1) == 0):
    raw_lidar_pcl = raw_lidar_pcl[np.sum(raw_lidar_pcl[:, :3],axis=1) != 0,:]

    # Translation from map to robot_footprint
    husky_footprint_coord = np.array([cfg.MAP_TO_VELO_X, cfg.MAP_TO_VELO_Y, cfg.MAP_TO_VELO_Z],
                                     dtype=np.float32)

    # Lidar points in map coordinate
    shifted_coord = raw_lidar_pcl[:, :3] + husky_footprint_coord

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

    # Raw scan within bounds
    raw_lidar_pcl = raw_lidar_pcl[bound_box]
    pcl_features = raw_lidar_pcl.copy()
    voxel_index = voxel_index[bound_box]
    if "vox_pos" in features:
        pcl_features[:, :3] = (pcl_features[:, :3] + husky_footprint_coord) - voxel_index * voxel_size

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    # Number of voxels in scan
    K = len(coordinate_buffer)
    # Max number of lidar points in each voxel
    T = cfg.VOXEL_POINT_COUNT

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=K, dtype=np.int64)

    # [K, T, 8] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 8), dtype=np.float32)
    label_buffer = np.zeros(shape=K, dtype=np.uint8)
    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, pcl_features):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :6] = point
            number_buffer[index] += 1
            labels,counts = np.unique(feature_buffer[index, :number_buffer[index], 5],return_counts=True)
            # Label is the maximum number of particle vs non particle
            label_buffer[index] = labels[np.argmax(counts)].astype(int)

    # Compute relative position
    for i in range(K):
        feature_buffer[i, :number_buffer[i], -3:] = feature_buffer[i, :number_buffer[i], :3] - \
                                                    feature_buffer[i, :number_buffer[i], :3].sum(axis=0,
                                                                                                 keepdims=True) / \
                                                    number_buffer[i]
    # Pick and choose features here
    selected_buffer = np.array([], dtype=np.float32).reshape(feature_buffer.shape[0], feature_buffer.shape[1], 0)

    # Create a mask to only populate number of points in the voxel and not all T points
    mask = ~np.all(feature_buffer[:, :, :] == 0, axis=2)
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

    if "pos" in features or "vox_pos" in features:
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 0:3]), axis=2)
    if "intensity" in features:
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 3:4]), axis=2)
        # Add Gaussian noise to intensity
        # selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:]+np.random.normal(0,3,(selected_buffer.shape[0],selected_buffer.shape[1],1)))*mask
    if "echo" in features:
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
    if not "echo" in features and "echo_one_hot" in features:
        # If one hot echo, concatenate three times to have three individual vectors for 0 1 2
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
        selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:] == 0) * mask
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
        selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:] == 1) * mask
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 4:5]), axis=2)
        selected_buffer[:, :, -1:] = (selected_buffer[:, :, -1:] == 2) * mask
    if "rel_pos" in features:
        selected_buffer = np.concatenate((selected_buffer, feature_buffer[:, :, 5:8]), axis=2)

    # Compute voxelised input for convolution (voxel_pos = true is necessary!)
    if "conv" in features:
        nb_subvoxel = cfg.NB_SUBVOXEL
        # initialize subvoxel for each feature
        pt_count_buffer = np.zeros(shape=(feature_buffer.shape[0], 1, pow(nb_subvoxel, 3)), dtype=np.float32)
        # intensity_buffer = np.zeros(shape=(feature_buffer.shape[0], 1, pow(nb_subvoxel, 3)), dtype=np.float32)
        # echo0_buffer = np.zeros(shape=(feature_buffer.shape[0], 1, pow(nb_subvoxel, 3)), dtype=np.float32)
        # echo1_buffer = np.zeros(shape=(feature_buffer.shape[0], 1, pow(nb_subvoxel, 3)), dtype=np.float32)
        # echo2_buffer = np.zeros(shape=(feature_buffer.shape[0], 1, pow(nb_subvoxel, 3)), dtype=np.float32)

        for idx in range(feature_buffer.shape[0]):
            # grab points in voxel (remove empty rows)
            voxel = selected_buffer[idx, ~np.all(feature_buffer[idx, :, :] == 0, axis=1), :]
            for row in range(voxel.shape[0]):
                # position index within voxel
                v_pos = np.floor(voxel[row, :3] / (cfg.VOXEL_X_SIZE / nb_subvoxel))
                voxel_id = int(v_pos[0] + v_pos[1] * nb_subvoxel + v_pos[2] * nb_subvoxel * nb_subvoxel)
                # Update subvoxels
                pt_count_buffer[idx, 0, voxel_id] = 1
                # intensity_buffer[idx, 0, voxel_id] += voxel[row, 3]
                # echo0_buffer[idx, 0, voxel_id] += voxel[row, 4]
                # echo1_buffer[idx, 0, voxel_id] += voxel[row, 5]
                # echo2_buffer[idx, 0, voxel_id] += voxel[row, 6]

            # Average values

            # Make a mask on subvoxels with no points to avoid division by zero
            # mask = (pt_count_buffer[idx, 0, :] == 0) * 1
            #
            # intensity_buffer[idx, 0, :] /= (pt_count_buffer[idx, 0, :] + mask)
            # echo0_buffer[idx, 0, :] /= (pt_count_buffer[idx, 0, :] + mask)
            # echo1_buffer[idx, 0, :] /= (pt_count_buffer[idx, 0, :] + mask)
            # echo2_buffer[idx, 0, :] /= (pt_count_buffer[idx, 0, :] + mask)

        # selected_buffer = np.concatenate(
        #     (pt_count_buffer, intensity_buffer, echo0_buffer, echo1_buffer, echo2_buffer), axis=2)
        selected_buffer = pt_count_buffer
    return selected_buffer, coordinate_buffer, label_buffer, raw_lidar_pcl
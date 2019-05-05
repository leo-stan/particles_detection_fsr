# Helper libraries
import numpy as np
import pypcd

folder_loc_leo = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/leo/"
folder_loc_julian = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/julian/"
folder_loc_general = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/general/"
target_loc = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/1_Upload/final/"

modify_leo = True
modify_julian = False # Only has an influence int he case where modify_leo is False
save_filtered_bag_as_npy = True # Only for modify_julian or general

for recording in [1,5,6,10,13,15,16,18]:
#for recording in [7,8,17]:
    if modify_leo:
        folder = folder_loc_leo + str(recording) + "/"
    elif modify_julian:
        folder = folder_loc_julian + str(recording) + "/"
    else:
        folder = folder_loc_general + str(recording) + "/"
    cloud_scans = np.asarray(np.load(folder + str(recording) + "_pred.npy"))
    # Target ------------------------------------------------------ (because we need very first scan)
    if recording < 9:
        cloud_target = np.load(target_loc + str(recording) + "-dust_labeled_spaces.npy")[0]
    else:
        cloud_target = np.load(target_loc + str(recording) + "-smoke_labeled_spaces.npy")[0]
    # Filter out dummy values
    cloud_target = cloud_target[((cloud_target[:, 0] != 0) | (cloud_target[:, 1] != 0)) | (cloud_target[:, 2] != 0)]

    cloud_target = cloud_target[:, :3]
    metadata_target = {"version": .7, "fields": ['x', 'y', 'z'], "size": [4, 4, 4], "count": [1, 1, 1],
                       "width": len(cloud_target), "height": 1, "viewpoint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       "points": len(cloud_target), "type": ['F', 'F', 'F'], "data": 'binary'}
    pc_target = pypcd.PointCloud(metadata_target, cloud_target)
    pc_target.save_pcd(folder + 'target.pcd')
    if save_filtered_bag_as_npy:
        cloud_save = []

    # Source ------------------------------------------------------
    for i in range(len(cloud_scans)):
        cloud_source = cloud_scans[i]
        cloud_source = cloud_source.astype(np.float32)
        # Filter out dummy values
        cloud_source = cloud_source[((cloud_source[:, 0] != 0) | (cloud_source[:, 1] != 0)) | (cloud_source[:, 2] != 0)]
        #print cloud_source[:,4]
        # Original ------------------------------------------------
        cloud_source_original = cloud_source[:, :3]

        metadata_source_original = {"version": .7, "fields": ['x','y','z'], "size": [4, 4, 4], "count": [1, 1, 1],
                   "width": len(cloud_source_original), "height": 1, "viewpoint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   "points": len(cloud_source_original), "type": ['F','F', 'F'], "data": 'binary'}
        pc_source_original = pypcd.PointCloud(metadata_source_original, cloud_source_original)
        pc_source_original.save_pcd(folder + 'original/' + str(i) + '.pcd')
        # Filtered ------------------------------------------------
        if modify_leo:
            print(cloud_source.shape)
            cloud_source_filtered = (cloud_source[(cloud_source[:, 6] == 0) & (cloud_source[:, 4] < 1.5)])[:, :3]
            print(cloud_source_filtered.shape)
        else:
            # Use non integer network predictions --> argmax
            print(cloud_source.shape)
            cloud_source_filtered = (cloud_source[np.argmax(cloud_source[:, 10:13], axis=1) == 0])[:, :3]
            print(cloud_source_filtered.shape)
        metadata_source_filtered = {"version": .7, "fields": ['x','y','z'], "size": [4, 4, 4], "count": [1, 1, 1],
                   "width": len(cloud_source_filtered), "height": 1, "viewpoint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   "points": len(cloud_source_filtered), "type": ['F','F', 'F'], "data": 'binary'}
        pc_source_filtered = pypcd.PointCloud(metadata_source_filtered, cloud_source_filtered)
        pc_source_filtered.save_pcd(folder + 'filtered/' + str(i) + '.pcd')
        # For saving later on
        if save_filtered_bag_as_npy:
            if modify_leo:
                cloud_save_local = np.zeros([len(cloud_source_filtered), 13])
                cloud_save_local[:,:3] = cloud_source_filtered[:,:3]
                #cloud_save_local[:,10] = 1 - cloud_source[:,6]
                cloud_save.append(cloud_save_local)
            else:
                indices = np.argmax(cloud_source[:, 10:13], axis=1)
                cloud_source[:,10:13] = 0
                for i in range(len(cloud_source)):
                    cloud_source[i,10+indices[i]] = 1
                cloud_save.append(cloud_source)
    if save_filtered_bag_as_npy:
        #cloud_save = np.asarray(cloud_save)
        np.save("/home/juli/Desktop/leo_filtered_pointclouds/" + str(recording) + "_pred.npy", cloud_save)

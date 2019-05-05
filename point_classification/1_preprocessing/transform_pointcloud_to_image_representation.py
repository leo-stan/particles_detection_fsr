# Same information as in the prepare_data_test jupyter notebook file

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Parameters you have to set manually
dual_pcl = True # Does it have 2 echo values?
global width
# For Calculation of width: width = number_points/per_ring = 360deg. / angular_resolution
width = 2172 # Corresponding to angular resolution of LiDAR for QUT dataset
# width = 3125 # Corresponding to angular resolution of KITTI dataset
#width = 1826 # Corresponding to ETH dataset
bool_self_detect_width = False

file_names = ["1-dust", "2-dust", "3-dust", "4-dust", "5-dust", "6-dust", "7-dust",
                  "8-dust", "9-smoke", "10-smoke", "11-smoke", "12-smoke", "13-smoke",
                  "14-smoke", "15-smoke", "16-smoke", "17-smoke", "18-smoke", "19-smoke"]
file_names = ["1_pred", "5_pred", "6_pred", "10_pred", "13_pred", "15_pred", "16_pred", "18_pred"]

path = "/home/juli/Desktop/icp_pred/"
path = "/media/juli/98F29C83F29C67721/SemesterProject/1_data/4_icp/general/"
path = "/home/juli/Downloads/"

def process_datasets(name):
    print name
    global width
    frames = np.load(name + ".npy")  # type: numpy file containing all of the labeled pointcloud points
    print frames.shape
    if bool_self_detect_width:
        width = 0
        for frame in frames:
            rings = frame[:, 4]
            for ring in np.unique(rings):
                width = np.fmax(width, len(frame[ring == rings]))
        print width

    rings = np.unique(frames[0, :, 4]) # Array which contains all rings values

    # Creates the "image representation" of the point clouds, also if the pointcloud is unordered
    # 15 channels will be written: d_1,x_1,y_1,z_1,i_1,r_1,d_2,x_2,y_2,z_2,i_2,r_2,l_none,l_dust,l_smoke
    if dual_pcl:
        images = np.zeros([len(frames), len(rings), width, frames[0].shape[1]+2], dtype=np.float32) # 2 times radius
    # 9 channels will be written: d_1,x_1,y_1,z_1,i_1,r_1,l_none,l_dust,l_smoke
    else:
        print len(frames), len(rings), width, frames[0].shape[1] + 1
        images = np.zeros([len(frames), len(rings), width, frames[0].shape[1] + 1], dtype=np.float32) # 1 time radius
    counter = 0
    for i, frame in enumerate(frames):
        print i
        counter += 1
        x_zero_indices = frame[:, 0] == 0 # Checks where x values are 0
        y_zero_indices = frame[:, 1] == 0 # Checks where y values are 0
        indices_equal = x_zero_indices == y_zero_indices # Checks where indices from before are equal
        indices = np.invert(x_zero_indices == indices_equal) # Produces inverted indices where x and y values are both 0
        frame = frame[indices] # Now only points which are not in the origin are remaining
        #if counter == 30:
           #break
        ring_values = frame[:,4] # Array of the ring of each point in the right order
        for ring in rings:
            ring_points = frame[ring==ring_values] # Get all points in current ring
            ring_angles = np.arctan2(ring_points[:,1], ring_points[:,0])
            ring_angles[ring_angles < 0] += 2*np.pi
            ring_indices = np.argsort(ring_angles)
            ring_angles = ring_angles[ring_indices] # Sorted Angle values
            ring_points = ring_points[ring_indices] # Sorted points in given ring (by angle)
            ring_pix_coord = ((width-1)*(ring_angles)/(2*np.pi))
            ring_relative_distances =  np.diff(ring_pix_coord)
            index = ring_pix_coord[0]
            old_index = 100000 # Dummy value for first step
            for k, pixel_point in enumerate(ring_points):
                if int(np.rint(index)) == int(np.rint(old_index)):
                    index = index + 1
                    print("Artifically jump 1 step")
                if int(np.rint(index)) >= width:
                    print("Ignored Points")
                    continue # In case of numerical errors (only once all 200 frames)
                images[i, int(np.rint(ring)), int(np.rint(index)), 0] = np.linalg.norm(pixel_point[:3])
                # old_index = int(np.rint(index))

                if dual_pcl:
                    images[i, int(np.rint(ring)), int(np.rint(index)), 1:6] = pixel_point[:5]
                    images[i, int(np.rint(ring)), int(np.rint(index)), 6] = np.linalg.norm(pixel_point[5:8])
                    images[i, int(np.rint(ring)), int(np.rint(index)), 7:] = pixel_point[5:]
                else:
                    images[i, int(np.rint(ring)), int(np.rint(index)), 1:] = pixel_point[:8]
                if k == len(ring_points)-1: # In this case we already reached last point --> distances one field smaller
                    continue
                old_index = index
                if k < width:
                    index = ring_pix_coord[k+1] # Not rounded here

    np.save(name + "_img.npy", images)

for name_start in file_names:
    name = path + name_start
    process_datasets(name)



    ''' # Working Block (Iteratively adds up the rounded indices)
    index = int(np.rint(ring_pix_coord[0]))
    # old_index = np.nan
    for k, pixel_point in enumerate(ring_points):
        if index >= width:
            print("Ignored Points")
            continue # In case of numerical errors (only once all 200 frames)
        if np.rint(index) == old_index:
            print("Write twice to same index for ring {}".format(ring))
         if np.rint(index) > old_index + 1:
            print("Jumped over one index")
         print(index - (np.rint(index)))
        images[i, int(np.rint(ring)), index, 0] = np.linalg.norm(pixel_point[:3])
        # old_index = int(np.rint(index))

        if dual_pcl:
            images[i, int(np.rint(ring)), index, 1:6] = pixel_point[:5]
            images[i, int(np.rint(ring)), index, 6] = np.linalg.norm(pixel_point[5:8])
            images[i, int(np.rint(ring)), index, 7:] = pixel_point[5:]
        else:
            images[i, int(np.rint(ring)), index, 1:] = pixel_point[:8]
        if k == len(ring_points)-1: # In this case we already reached last point --> distances one field smaller
            continue
        index += int(np.rint(ring_relative_distances[k]))
    
    '''
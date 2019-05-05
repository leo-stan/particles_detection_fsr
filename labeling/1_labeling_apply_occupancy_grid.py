# Initialization
import numpy as np
# Test
from matplotlib import pyplot as plt
from IPython.display import clear_output
from subprocess import Popen
import sys
import gc
import time
from joblib import Parallel, delayed
import multiprocessing

# Assumption: Input pcl has 5 values (x, y, z, intensity, ring) per point if single_pcl, 10 otherwise

QUT_bag = True # True if it's one of the QUT bags, False otherwise
                # This is also equivalent to having same amount of points for each ring (easier iterable)
dual_point_cloud = True
file_names = ["/1-dust", "/2-dust", "/3-dust", "/4-dust", "/5-dust", "/6-dust", "/7-dust", "/8-dust", "/9-smoke",
              "/10-smoke", "/11-smoke", "/12-smoke", "/13-smoke", # QUT bags
              "/14-smoke", "/15-smoke", "/16-smoke", "/17-smoke", "/18-smoke", "/19-smoke"] #,
#              "/20-smoke", "/21-smoke", "/22-smoke", "/23-smoke", "/24-smoke", "/25-smoke"]
file_names = file_names[::-1]
print(file_names)
#file_names = ["/2-dust"]
path = "/media/juli/98F29C83F29C67721/SemesterProject/data/Upload/1_train_val_set/numpy_files"
#path = "/home/juli/Downloads"
#file_names = ["/test"]
amount_labels = 3 # label_none, label_dust, label_smoke

# Bag characteristics
amount_rings = 0
max_num_pts_per_ring = 0


# Data manipulation functions
def correct_frames(frames):
    corrected_frames = []
    width = max_num_pts_per_ring
    for i, points in enumerate(frames):
        # print("Iter i:{}".format(i))
        ring = points[:, 4]
        # print max(frames[0][:,4]), min(frames[0][:,4])
        # print max(ring), min(ring)
        # print("Rings are:{}".format(ring))
        grid = []
        for n in np.unique(ring): # Also sorts it depending on rings
            # print n
            ring_points = points[n == ring]
            if QUT_bag: # "QUT" in load_file:  # Only for this specific Velodyne (if each ring has same no. of points)
                grid.append(ring_points)
            else:
                ring_angles = np.arctan2(ring_points[:, 1], ring_points[:, 0])
                ring_angles[ring_angles < 0] += 2 * np.pi
                ring_indices = np.argsort(ring_angles)
                ring_angles = ring_angles[ring_indices]  # Sorted Angle values
                ring_points = ring_points[ring_indices]  # Sorted points in given ring (by angle)
                ring_pix_coord = ((width - 1) * (ring_angles) / (2 * np.pi))
                ring_relative_distances = np.diff(ring_pix_coord)
                image = np.zeros([width, 5])
                image[:,4] = n # Fill ring value correctly
                index = ring_pix_coord[0] # Round it later to avoid accumulating rounding  errors
                counter = 0
                for k, pixel_point in enumerate(ring_points):
                    if index >= width:
                        counter += 1
                        continue  # In case of numerical errors (only once all 200 frames)
                    image[int(np.rint(index))] = pixel_point
                    if k == len(ring_points) - 1:  # In this case we already reached last point --> distances one field smaller
                        continue
                    index += ring_relative_distances[k]
                grid.append(image)
        corrected_frames.append(grid)
    return np.asarray(corrected_frames)



# Get Information from given ring out of given frame and calculate log-radius and angle
def get_ring_i(ring_id, scan_id, frames):
    ring_0 = frames[scan_id][ring_id]

    x0 = ring_0[:, 0]
    y0 = ring_0[:, 1]
    z0 = ring_0[:, 2]
    i0 = ring_0[:, 3]
    index = ring_0[:,5]

    w0 = np.arctan2(y0, x0)
    r0 = np.sqrt(x0 * x0 + y0 * y0)
    r0_log = np.log10(r0 + 1)

    return x0, y0, z0, w0, r0_log, i0, index


# Create the given occupancy grid
def occupancy(w, r_log, bool_static):
    #     oc_grid = np.zeros((901, 200))
    oc_grid = np.zeros((4000, 1500), dtype=np.float32) # Completely overdimensioned
    angle_bin = np.rint(w * 57.2958 / 0.4) + 450  # +450 # 180/pi = 57.2958 --> Mapped to values from 0 to 900
    distance_bin = np.rint(r_log / 0.05)

    # print("Angle bin:{}".format(angle_bin))
    # print("len angle bin:{}".format(len(angle_bin)))

    angle_bin = angle_bin.astype(int)
    distance_bin = distance_bin.astype(int)

    # print angle_bin
    # print distance_bin

    # print("Range of angle bins:{}:{}".format(max(angle_bin), min(angle_bin)))
    # print("Range of distance bins:{}:{}".format(max(distance_bin), min(distance_bin)))

    # print oc_grid
    for i in range(len(angle_bin)):
        if bool_static and angle_bin[i] > 0 and distance_bin[i] > 0: # For robustness: Mark surrounding entries as occupied (e.g. when point is close to cell boundary)
            for angle_counter in range(-4,5):
                for distance_counter in range(-1,2):
                    oc_grid[angle_bin[i]+angle_counter, distance_bin[i]+distance_counter] += 1
        else:
            oc_grid[angle_bin[i], distance_bin[i]] += 1

    # print oc_grid

    return oc_grid


# Subtract two voxel grids from each other
def groundtruth_ring(static, smoky):
    gt = smoky - static
    r, c = np.where(gt < 1)
    gt[r, c] = 0
    return gt


#
def unpack_gt(gt, x_smoky, y_smoky, z_smoky, w_smoky, r_smoky, i_smoky, index_smoky):
    x_gt, y_gt, z_gt, w_gt, r_gt, label_gt, i_gt, index_gt = [], [], [], [], [], [], [], []

    angle_bin = np.rint(w_smoky * 57.2958 / 0.4) + 450 # project [-pi,pi] to [0,900]
    distance_bin = np.rint(r_smoky / 0.05)

    angle_bin = angle_bin.astype(int)
    distance_bin = distance_bin.astype(int)
    # print angle_bin
    # print distance_bin

    # print len(angle_bin), len(distance_bin)

    r, c = np.where(gt >= 1)  # For accum geq, for single maybe >
    for i in range(len(angle_bin)):
        if distance_bin[i] == 0:  # Afterwards added values (not really existent)
            x_gt.append(x_smoky[i])
            y_gt.append(y_smoky[i])
            z_gt.append(z_smoky[i])
            w_gt.append(w_smoky[i])
            r_gt.append(r_smoky[i])
            label_gt.append(0)  # Changed from 2 to 0
            i_gt.append(i_smoky[i])
            index_gt.append(index_smoky[i])
        # elif angle_bin[i] in r and distance_bin[i] in c: # false in my opinion
        elif distance_bin[i] in np.array(c)[np.array(r) == angle_bin[i]]:  # Corrected
            x_gt.append(x_smoky[i])
            y_gt.append(y_smoky[i])
            z_gt.append(z_smoky[i])
            w_gt.append(w_smoky[i])
            r_gt.append(r_smoky[i])
            label_gt.append(1)
            i_gt.append(i_smoky[i])
            index_gt.append(index_smoky[i])
        else:
            x_gt.append(x_smoky[i])
            y_gt.append(y_smoky[i])
            z_gt.append(z_smoky[i])
            w_gt.append(w_smoky[i])
            r_gt.append(r_smoky[i])
            label_gt.append(0)
            i_gt.append(i_smoky[i])
            index_gt.append(index_smoky[i])
    return x_gt, y_gt, z_gt, w_gt, r_gt, label_gt, i_gt, index_gt


# Use multiple frames in the beginning as ground truth
def generate_file_accumulated(frames, number_static_frames):
    image_width = max_num_pts_per_ring

    rings_present = np.arange(amount_rings)
    print("rings_present:{}".format(rings_present))
    static_scene = []
    for ring_id in rings_present:
        x0, y0, z0, w0, r0_log, i0, index0 = [], [], [], [], [], [], []
        for i in range(number_static_frames):
            x0_dyn, y0_dyn, z0_dyn, w0_dyn, r0_log_dyn, i0_dyn, index0_dyn = get_ring_i(ring_id, i, frames)
            x0 = np.concatenate((x0, x0_dyn))
            y0 = np.concatenate((y0, y0_dyn))
            z0 = np.concatenate((z0, z0_dyn))
            w0 = np.concatenate((w0, w0_dyn))
            r0_log = np.concatenate((r0_log, r0_log_dyn))
            i0 = np.concatenate((i0, i0_dyn))
            index0 = np.concatenate((index0, index0_dyn))
        static_scene.append([x0, y0, z0, i0, w0, r0_log, index0])
    # print static_scene[10].keys()
    labeled_frames = np.zeros((len(frames), len(rings_present), image_width, 7), dtype=np.float32)

    lock = multiprocessing.Lock()
    def process_frame(frame_id):
        print frame_id
        dynamic_scene = np.zeros((len(rings_present), image_width, 7), dtype=np.float32)
        for ring_id in rings_present:
            temporal_scene = np.zeros((image_width, 7), dtype=np.float32)
            x1, y1, z1, w1, r1_log, i1, index1 = get_ring_i(ring_id, frame_id, frames)
            oc_grid_static = occupancy(static_scene[ring_id][4], static_scene[ring_id][5], True) # Take out angles and distances
            oc_grid_smoky = occupancy(w1, r1_log, False)

            gt_grid = groundtruth_ring(oc_grid_static, oc_grid_smoky)
            x_gt, y_gt, z_gt, w_gt, r_gt, label_gt, i_gt, index_gt = unpack_gt(gt_grid, x1, y1, z1, w1, r1_log, i1, index1)
            temporal_scene[:, 0] = x_gt[:image_width]
            temporal_scene[:, 1] = y_gt[:image_width]
            temporal_scene[:, 2] = z_gt[:image_width]
            temporal_scene[:, 3] = i_gt[:image_width]
            temporal_scene[:, 4] = ring_id
            temporal_scene[:, 5] = label_gt[:image_width]
            temporal_scene[:, 6] = index_gt[:image_width]
            dynamic_scene[ring_id, :, :] = temporal_scene

        return dynamic_scene

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(process_frame)(frame_id) for frame_id in range(len(frames)))
    for frame_id in range(len(frames)):
        labeled_frames[frame_id] = results[frame_id]
    print(labeled_frames.shape)
    return labeled_frames

for name in file_names:
    print(name)
    frames = np.load(path + name + "_labeled_spaces.npy")
    #frames = frames[:250]
    # frames = frames[:50]
    # Calculate characteristics of bag
    example_frame = frames[0]
    amount_rings = len(np.unique(example_frame[:, 4]))
    del example_frame
    print amount_rings
    max_num_pts_per_ring = 0
    for frame in frames:
        rings = frame[:, 4]
        for ring in np.unique(rings):
            max_num_pts_per_ring = np.fmax(max_num_pts_per_ring, len(frame[ring == rings]))
    print max_num_pts_per_ring

    if dual_point_cloud:
        frames_2nd_echo = frames[:,:,:5]
        new_frames = np.zeros([frames.shape[0],frames.shape[1], 6]) # 6 because we then have 1 entry for later synchronization
        new_frames[:,:,:5] = frames[:, :, 5:10]
        # For later syncing with dual point cloud
        for i in range(len(new_frames)):
            new_frames[i,:,5] = range(new_frames.shape[1])
        frames = new_frames
    else:
        frames_2nd_echo = [[[]]]
        for i, frame in enumerate(frames):
            new_frame = np.zeros([frame.shape[0],6])
            new_frame[:,:5] = frame
            new_frame[:,5] = range(new_frame.shape[0])
            frames[i] = new_frame
    corrected_frames = correct_frames(frames)
    del frames
    labeled_frames = generate_file_accumulated(corrected_frames, 10)
    del corrected_frames
    # Bring them back to original form (without rings as own dimension)
    frame_ref = labeled_frames[0]
    number_frames = len(labeled_frames)
    number_points = len(frame_ref[0])
    print(number_points)
    number_rings = len(frame_ref)
    print(number_rings)
    number_entries = len(frame_ref[0][0])
    print(number_entries)
    interm_frames = np.zeros([number_frames, number_points * number_rings, number_entries], dtype=np.float32)
    for j, frame in enumerate(labeled_frames):
        write_frames = np.zeros((number_points * number_rings, number_entries))
        for i in range(number_points):
            for ring in range(number_rings):
                write_frames[ring + i * number_rings] = frame[ring][i]
        interm_frames[j,:,:] = write_frames # Has 1 value for the labels
    del labeled_frames
    del frame_ref
    gc.collect()
    print("sleeps...")
    time.sleep(5)
    print("returns...")
    # Put everything together
    # Synchronize points so that same angle and ring for first and second echo
    new_frames = np.zeros([interm_frames.shape[0], interm_frames.shape[1], 6])
    for i in range(len(new_frames)):
        args = np.argsort(interm_frames[i,:,6])
        new_frames[i, :, :] = interm_frames[i, args, :6]
    interm_frames = new_frames
    del new_frames

    print(len(interm_frames[0][0]) - 1 + len(frames_2nd_echo[0][0]) + amount_labels) # -1 since don't want labels in width
    # From here on adapt it to multiclass labeling
    labels = interm_frames[:,:,len(interm_frames[0][0]) - 1]
    final_frames = np.zeros([len(interm_frames), len(interm_frames[0]),
                            len(interm_frames[0][0]) - 1 + len(frames_2nd_echo[0][0]) + amount_labels], dtype=np.float32)
    final_frames[:,:,:len(interm_frames[0][0])-1] = interm_frames[:,:,:len(interm_frames[0][0])-1]
    final_frames[:,:,len(interm_frames[0][0])-1:len(interm_frames[0][0])-1 + len(frames_2nd_echo[0][0])] = frames_2nd_echo
    final_frames[:, :, len(interm_frames[0][0])-1 + len(frames_2nd_echo[0][0])] = 1 - labels
    if "dust" in name:
        final_frames[:,:,len(interm_frames[0][0]) + len(frames_2nd_echo[0][0])] = labels
    else: # for everything else assume smoke so far (e.g. KITTI)
        final_frames[:, :, len(interm_frames[0][0]) + len(frames_2nd_echo[0][0]) + 1] = labels
    np.save(path + name + "_labeled.npy", final_frames)
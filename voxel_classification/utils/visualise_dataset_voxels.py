import numpy as np
import os.path as osp
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
import sys
sys.path.append('../src')
from config import cfg
import yaml

# Load data
dataset = 'use_case'
dataset_dir = osp.join(cfg.DATASETS_DIR, dataset)

with open(osp.join(dataset_dir, 'config.yaml'), 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

rospy.init_node('visualise_labeled_lidar_pcl', anonymous=True)
bridge = CvBridge()

header = Header()
header.frame_id = 'voxel_map'

lidar_pub = rospy.Publisher("velodyne_voxels_labeled", Pc2, queue_size=1)  # Declare publisher

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)
p_label = Pf('label', 14, 2, 1)
fields = [p_x, p_y, p_z, p_label]

rate = rospy.Rate(10)

for i in range(params['nb_scans']):
    voxel_label = np.load(osp.join(dataset_dir, 'scan_voxels', 'labels_' + str(i) + '.npy'))
    voxel_coord = np.load(osp.join(dataset_dir, 'scan_voxels', 'coords_' + str(i) + '.npy'))

    header.stamp = rospy.Time.now()
    lidar_pub.publish(p_c2.create_cloud(header, fields, np.concatenate(
        (voxel_coord * np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE]), voxel_label.reshape(-1, 1)),
        axis=1)))
    rate.sleep()
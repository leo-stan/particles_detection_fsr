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
header.frame_id = 'velodyne'

lidar_pub = rospy.Publisher("velodyne_points_labeled", Pc2, queue_size=1)  # Declare publisher

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)
p_int = Pf('intensity', 12, 2, 1)
p_echo = Pf('echo', 13, 2, 1)
p_gtlabel = Pf('gt_label', 14, 2, 1)
fields = [p_x, p_y, p_z, p_int, p_echo, p_gtlabel]

rate = rospy.Rate(10)

for i in range(params['nb_scans']):
    pcl = np.load(osp.join(dataset_dir, 'scan_pcls', str(i) + '.npy'))

    header.stamp = rospy.Time.now()
    lidar_pub.publish(p_c2.create_cloud(header, fields, pcl))
    rate.sleep()
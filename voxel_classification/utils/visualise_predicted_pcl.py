import numpy as np
import os.path as osp
import os
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as Pc2
import sensor_msgs.point_cloud2 as p_c2
from sensor_msgs.msg import PointField as Pf
import sys
sys.path.append('../src')
from config import cfg

# Load data
model = '20190429_152306_test_model'
eval_name = 'eval_20190429_160431_val'
path = os.path.join(cfg.LOG_DIR,model,'evaluations',eval_name)
rospy.init_node('visualise_labeled_lidar_pcl', anonymous=True)
bridge = CvBridge()

header = Header()
header.frame_id = 'velodyne'

lidar_pub = rospy.Publisher("velodyne_points_predicted", Pc2, queue_size=1)  # Declare publisher

p_x = Pf('x', 0, 7, 1)
p_y = Pf('y', 4, 7, 1)
p_z = Pf('z', 8, 7, 1)
p_int = Pf('intensity', 12, 2, 1)
p_echo = Pf('echo', 13, 2, 1)
p_gtlabel = Pf('gt_label', 14, 2, 1)
p_label = Pf('pred_label', 15, 2, 1)
fields = [p_x, p_y, p_z,p_int,p_echo,p_gtlabel,p_label]

rate = rospy.Rate(20)

# for f in filename:
print("loading file...")
pcl = np.load(osp.join(path,'scans.npy'))
print("displaying file...")
for p in pcl:
    header.stamp = rospy.Time.now()
    lidar_pub.publish(p_c2.create_cloud(header, fields, p))
    rate.sleep()
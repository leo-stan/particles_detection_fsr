# Writing rosbags to .npy files

For this purpose I always used either the file store_single_rosbag.ipynb (if your pointcloud has only one echo) or store_multiple_rosbags if you want to extract multiple topics.

I tried to automate this writing process but due to issues with opening several separated subscribers this didn't work.

THEREFORE CURRENTLY you need to restart the kernel after each file written. 

# Visualize the pointcloud in rviz

For this the file publish_pointcloud.ipynb can be used. The use is quite starightforward. First load the desired file and then a rostopic will be published. For this a roscore needs to run. 

# Writing to rosbag

If you would like to write a -npy pointcloud again to a rosbag you can use the write_rosbag.ipynb file.

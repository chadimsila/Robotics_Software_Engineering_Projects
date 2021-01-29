#!/usr/bin/env python

# Import modules

import rospy
import pcl
from pcl_helper import *
import numpy as np
import ctypes
import struct
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from random import randint


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # TODO: Initialization
    cloud=pc2.read_points(pcl_msg)
    l[]
    for i in cloud:
        l.append([[i[0],i[1],i[2]]])

    pcl_point=pcl.P

    # TODO: Convert ROS msg to PCL data

    # TODO: Voxel Grid Downsampling

    # TODO: PassThrough Filter


    # TODO: RANSAC Plane Segmentation

    # TODO: Extract inliers and outliers

    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages

    # TODO: Publish ROS messages


if __name__ == '__main__':

    # TODO: ROS node initialization

    # TODO: Create Subscribers

    # TODO: Create Publishers

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
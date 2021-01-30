#!/usr/bin/env python

# Import modules


import pcl
from sensor_msgs.msg import PointCloud2, PointField


def ros_to_pcl(ros_cloud):

    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # TODO: Initialization

    # TODO: Convert ROS msg to PCL data
    point_cloud=ros_to_pcl(ros_cloud)
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
    rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    # Initialize color_list

    # TODO: Spin while node is not shutdown
        while not rospy.is_shutdown():
            rospy.spin()

    # TODO: Publish ROS msg
    pcl_objects_pub.publish(pcl_msg)
    pcl_table_pub.publish(pcl_msg)
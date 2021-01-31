#!/usr/bin/env python
# Import modules
from pcl_helper import *

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # TODO: Initialization
    
    # TODO: Convert ROS msg to PCL data
    cloud=ros_to_pcl(pcl_msg)
    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()  
    LEAF_SIZE = 0.01  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name("z")
    passthrough.set_filter_limits(0.6, 1.1)
    cloud_filtered = passthrough.filter()
    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)    
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    
    # TODO: Convert PCL data to ROS messages
    ros_table=pcl_to_ros(extracted_inliers)
    ros_object=pcl_to_ros(extracted_outliers)
    
    pcl_objects_pub.publish(ros_object)
    pcl_table_pub.publish(ros_table) 


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

#ifndef _TUG_G2O_BASED_MAPPING__G2O_BASED_MAPPING_HPP_
#define _TUG_G2O_BASED_MAPPING__G2O_BASED_MAPPING_HPP_

#include "Eigen/Dense"

#include "g2o/core/hyper_graph.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/data/laser_parameters.h"
#include "g2o/types/data/raw_laser.h"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace tug_g2o_based_mapping
{

class G2OBasedMapping : public rclcpp::Node 
{
  // Directives ----------------------------------------------------------------
  private:
    using Point = geometry_msgs::msg::Point;
    using Point32 = geometry_msgs::msg::Point32;
    using PoseWithCovarianceStamped
      = geometry_msgs::msg::PoseWithCovarianceStamped;
    using Quaternion = geometry_msgs::msg::Quaternion;
    using TransformStamped = geometry_msgs::msg::TransformStamped;

    using OccupancyGrid = nav_msgs::msg::OccupancyGrid;
    using Odometry = nav_msgs::msg::Odometry;

    using LaserScan = sensor_msgs::msg::LaserScan;
    using PointCloud = sensor_msgs::msg::PointCloud;

    using Marker = visualization_msgs::msg::Marker;
    using MarkerArray = visualization_msgs::msg::MarkerArray;

  // Variables -----------------------------------------------------------------
  private:
    // TF2
    std::shared_ptr<tf2_ros::Buffer> transform_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> transform_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
    
    // Map publish timer
    rclcpp::CallbackGroup::SharedPtr map_publish_timer_callback_group_;
    rclcpp::TimerBase::SharedPtr map_publish_timer_;

    // Publisher
    rclcpp::Publisher<OccupancyGrid>::SharedPtr map_publisher_;
    rclcpp::Publisher<PointCloud>::SharedPtr graph_cloud_publisher_;
    rclcpp::Publisher<MarkerArray>::SharedPtr robot_pose_marker_publisher_;
    rclcpp::Publisher<MarkerArray>::SharedPtr
      fiducials_observed_marker_publisher_;
    rclcpp::Publisher<Marker>::SharedPtr graph_edges_publisher_;
    rclcpp::Publisher<MarkerArray>::SharedPtr
      old_fiducials_observed_marker_publisher_;

    // Subscriber
    rclcpp::Subscription<Odometry>::SharedPtr odometry_subscriber_;
    rclcpp::Subscription<LaserScan>::SharedPtr laser_scan_subscriber_;
    rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr
      initial_pose_subscriber_;

    // Mutex
    std::mutex data_mutex_;

    // Messages
    Odometry last_odometry_;
    OccupancyGrid graph_map_;

    // Matrices
    Eigen::MatrixXd x_;
    Eigen::Matrix3d odom_noise_;
    Eigen::Matrix3d laser_noise_;
    Eigen::Matrix2d landmark_noise_;

    // Solver
    g2o::OptimizationAlgorithmGaussNewton* optimization_algorithm_;
    g2o::SparseOptimizer graph_;
    g2o::HyperGraph::VertexSet vertex_set_;
    g2o::HyperGraph::EdgeSet edge_set_;
    g2o::LaserParameters* laser_params_;

    // Misc
    bool reset_;
    bool valid_;
    bool first_opt_;
    bool robot_pose_set_;

    int min_to_optimize_;
    int last_id_;

    std::vector<int> seen_landmarks_;
    std::vector<int> robot_pose_ids_;
    std::vector<std::pair<int, int> > robot_landmark_edge_ids_;
    std::vector<std::pair<int, int> > laser_edge_ids_;

  // Methods -------------------------------------------------------------------
  public:
    G2OBasedMapping();
    ~G2OBasedMapping();

  private:
    void updateOdometry(const Odometry::ConstSharedPtr& odom);
    void updateLaser(const LaserScan::ConstSharedPtr& laser);
    void init(double x, double y, double theta);

    // adds odom vertex
    // takes global position and a unique id, first only needed in init
    void addOdomVertex(
      double x,
      double y,
      double theta,
      int id,
      bool first = false
    );
    
    // adds laser vertex
    // takes global position, a laser scan and a unique id,
    // first only needed in init
    void addLaserVertex(
      double x,
      double y,
      double theta,
      LaserScan scan,
      int id,
      bool first = false
    );

    // adds edge between the vertices based on a 2D transformation
    void addLaserEdge(
      int id1,
      int id2,
      double x,
      double y,
      double yaw,
      Eigen::Matrix3d noise
    );

    // adds edge between the vertices
    // takes two ids of odom vertices
    void addOdomEdge(int id1, int id2);
    
    // adds landmark vertex
    // takes global postion of the landmark and the corresponding id
    void addLandmarkVertex(double x, double y, int id);
  
    // adds landmark edge
    // takes odom vertex and landmark id and the global position of the landmark
    void addLandmarkEdge(int id1, int id2, double x, double y);
    
    // optimize graph
    void optimizeGraph();
    
    // resets the robot position to the odom vertex with the given id
    // takes id of odom vertex 
    void setRobotToVertex(int id);

    // helper to bridge g2o with ros
    LaserScan rawLasertoLaserScanMsg(g2o::RawLaser rawlaser);

    void visualizeLaserScans();
    void visualizeRobotPoses();
    void visualizeLandmarks();
    void visualizeEdges();
    void visualizeOldLandmarks();
    void updateLocalization();
    void publishMap();

    double yawFromQuaternion(const Quaternion& quaternion);

    // Subscriber callbacks
    void odomCallback(const Odometry::ConstSharedPtr& msg);
    void laserCallback(const LaserScan::ConstSharedPtr& msg);
    void initialPoseCallback(
      const PoseWithCovarianceStamped::ConstSharedPtr& msg
    );

    bool detectAndAddLoopClosures(int id);
    bool addScanMatchingEdge(int id1, int id2);
};

} /* namespace tug_g2o_based_mapping */

#endif /* _TUG_G2O_BASED_MAPPING__G2O_BASED_MAPPING_HPP_ */
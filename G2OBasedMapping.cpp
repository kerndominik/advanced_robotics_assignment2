#include "tug_g2o_based_mapping/G2OBasedMapping.hpp"

#include <chrono>
#include <functional>
#include <mutex>

#include "g2o/core/block_solver.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/types/slam2d/vertex_point_xy.h"
#include "g2o/types/slam2d/edge_se2_pointxy.h"
#include "g2o/types/slam2d/vertex_se2.h"
#include "g2o/types/slam2d/edge_se2.h"

#include "tf2/LinearMath/Matrix3x3.hpp"
#include "tf2/LinearMath/Quaternion.hpp"

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// Parameters
// Optimization parameters
#define OPTIMIZATION_ITERATIONS 10
#define MIN_TO_OPTIMIZE 4
// Noise parameters 
#define X_NOISE 0.2
#define Y_NOISE 0.2
#define ROT_NOISE 0.2
#define LASER_NOISE 1000
// ICP parameters
#define ICP_MAX_ITERATIONS 10
#define ICP_DIST_THRESHOLD 1.0
// Parameters for adding vertices
#define TRANS_THRESHOLD 0.3  // meters
#define ROT_THRESHOLD 0.3  // radians
// Loop closure parameters
#define FIND_LOOP_CLOSURE true
#define LOOP_CLOSURE_SEARCH_RADIUS TRANS_THRESHOLD  // meters
#define LOOP_CLOSURE_MIN_ID_DIFF 5
#define MAX_CLOSURES_PER_NODE 2

namespace tug_g2o_based_mapping
{

// -----------------------------------------------------------------------------
G2OBasedMapping::G2OBasedMapping()
  : Node("tug_g2o_based_mapping")
{
  // TF2
  transform_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(
    *transform_buffer_
  );

  transform_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(
    *this
  );

  // Map publish timer
  map_publish_timer_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive
  );

  map_publish_timer_ = rclcpp::create_timer(
    this,
    get_clock(),
    std::chrono::milliseconds(1000), // Update rate may be modified
    std::bind(&G2OBasedMapping::publishMap, this),
    map_publish_timer_callback_group_
  );

  // Publisher
  map_publisher_ = create_publisher<OccupancyGrid>("map", 10);
  graph_cloud_publisher_ = create_publisher<PointCloud>("graph_cloud", 10);
  robot_pose_marker_publisher_ = create_publisher<MarkerArray>(
    "robot_pose_marker",
    10
  );

  fiducials_observed_marker_publisher_ = create_publisher<MarkerArray>(
    "fiducials_observed_marker",
    10
  );

  graph_edges_publisher_ = create_publisher<Marker>("graph_edges", 10);
  old_fiducials_observed_marker_publisher_ = create_publisher<MarkerArray>(
    "old_fiducials_observed_marker",
    10
  );

  // Subscriber
  odometry_subscriber_ = create_subscription<Odometry>(
    "odometry",
    100,
    std::bind(&G2OBasedMapping::odomCallback, this, std::placeholders::_1)
  );

  laser_scan_subscriber_ = create_subscription<LaserScan>(
    "scan",
    100,
    std::bind(&G2OBasedMapping::laserCallback, this, std::placeholders::_1)
  );

  initial_pose_subscriber_ = create_subscription<PoseWithCovarianceStamped>(
    "initialpose",
    100,
    std::bind(
      &G2OBasedMapping::initialPoseCallback,
      this,
      std::placeholders::_1
    )
  );

  // Messages
  graph_map_.header.frame_id = "map";
  graph_map_.info.map_load_time = get_clock()->now();
  graph_map_.info.origin.position.x = -15.0;
  graph_map_.info.origin.position.y = -15.0;
  graph_map_.info.resolution = 0.05F;
  graph_map_.info.width = 600;
  graph_map_.info.height = 600;

  // Matrices
  x_ = Eigen::MatrixXd::Zero(3, 1);

  // Solver
  using SLAMLinearSolver 
    = g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>;
  
  std::unique_ptr<SLAMLinearSolver> solver
    = std::make_unique<SLAMLinearSolver>();

  solver->setBlockOrdering(false);

  optimization_algorithm_ = new g2o::OptimizationAlgorithmGaussNewton(
    std::make_unique<g2o::BlockSolverX>(std::move(solver))
  );

  graph_.setAlgorithm(optimization_algorithm_);

  // Init
  init(0.0, 0.0, 0.0);

  // TODO
  // find appropriate parameters
  double x_noise = X_NOISE;
  double y_noise = Y_NOISE;
  double rot_noise = ROT_NOISE;    //rad
  double landmark_x_noise = 1;
  double landmark_y_noise = 1;

  odom_noise_.fill(0.0);
  odom_noise_(0, 0) = 1 / (x_noise * x_noise);
  odom_noise_(1, 1) = 1 / (y_noise * y_noise);
  odom_noise_(2, 2) = 1 / (rot_noise * rot_noise);

  laser_noise_.fill(0.0);
  laser_noise_(0, 0) = LASER_NOISE;
  laser_noise_(1, 1) = LASER_NOISE;
  laser_noise_(2, 2) = LASER_NOISE;

  landmark_noise_.fill(0.0);
  landmark_noise_(0, 0) = 1 / (landmark_x_noise * landmark_x_noise);
  landmark_noise_(1, 1) = 1 / (landmark_y_noise * landmark_y_noise);
}

// -----------------------------------------------------------------------------
G2OBasedMapping::~G2OBasedMapping()
{
  delete optimization_algorithm_;
  delete laser_params_;
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::updateOdometry(const Odometry::ConstSharedPtr& odom)
{
  if (reset_)
  {
    last_odometry_ = *odom;

    updateLocalization();

    double x = odom->pose.pose.position.x;
    double y = odom->pose.pose.position.y;
    double yaw = yawFromQuaternion(odom->pose.pose.orientation);

    init(x, y, yaw);

    reset_ = false;
    valid_ = false;
    return;
  }

  double x_curr = odom->pose.pose.position.x;
  double y_curr = odom->pose.pose.position.y;
  double yaw_curr = yawFromQuaternion(odom->pose.pose.orientation);

  double x_last = last_odometry_.pose.pose.position.x;
  double y_last = last_odometry_.pose.pose.position.y;
  double yaw_last = yawFromQuaternion(last_odometry_.pose.pose.orientation);

  double dx = x_curr - x_last;
  double dy = y_curr - y_last;
  double dtheta = atan2(sin(yaw_curr - yaw_last), cos(yaw_curr - yaw_last));

  double dx_robot =  cos(yaw_last) * dx + sin(yaw_last) * dy;
  double dy_robot = -sin(yaw_last) * dx + cos(yaw_last) * dy;
  double theta = x_(2);

  x_(0) += cos(theta) * dx_robot - sin(theta) * dy_robot;
  x_(1) += sin(theta) * dx_robot + cos(theta) * dy_robot;
  x_(2) += dtheta;
  x_(2) = atan2(sin(x_(2)), cos(x_(2)));

  last_odometry_ = *odom;
}

std::vector<Eigen::Vector2d> laserToPoints(
  const g2o::RawLaser* raw,
  const std::vector<double>& pose = std::vector<double>(3, 0.0))
{
  std::vector<Eigen::Vector2d> points;
  double x = pose.at(0);
  double y = pose.at(1);
  double theta = pose.at(2);
  double angle = raw->laserParams().firstBeamAngle;
  for (double r : raw->ranges())
  {
    if (r <= raw->laserParams().minRange || r > raw->laserParams().maxRange)
    {
      angle += raw->laserParams().angularStep;
      continue;
    }
    double lx = r * cos(angle);
    double ly = r * sin(angle);
    // transform to world frame
    double wx = x + lx * cos(theta) - ly * sin(theta);
    double wy = y + lx * sin(theta) + ly * cos(theta);

    points.emplace_back(wx, wy);
    angle += raw->laserParams().angularStep;
  }
  return points;
}

bool computeICP(
  const std::vector<Eigen::Vector2d>& src,
  const std::vector<Eigen::Vector2d>& dst,
  double& dx,
  double& dy,
  double& dtheta,
  double guess_dx = 0.0,
  double guess_dy = 0.0,
  double guess_dtheta = 0.0
)
{
  if (src.empty() || dst.empty())
    return false;

  Eigen::Matrix2d R;
  R << cos(guess_dtheta), -sin(guess_dtheta),
       sin(guess_dtheta),  cos(guess_dtheta);
       
  Eigen::Vector2d T(guess_dx, guess_dy);

  std::vector<Eigen::Vector2d> src_transformed = src;

  for (auto& p : src_transformed)
    p = R * p + T;

  for (int iter = 0; iter < ICP_MAX_ITERATIONS; iter++)
  {
    std::vector<Eigen::Vector2d> src_corr;
    std::vector<Eigen::Vector2d> dst_corr;

    // --- Nearest neighbor ---
    for (const auto& p : src_transformed)
    {
      double min_dist = ICP_DIST_THRESHOLD;
      Eigen::Vector2d best_q;

      for (const auto& q : dst)
      {
        double dist = sqrt((p - q).squaredNorm());
        if (dist < min_dist)
        {
          min_dist = dist;
          best_q = q;
        }
      }
      // optionally reject outliers based on distance
      if (min_dist < ICP_DIST_THRESHOLD) // reject outliers
      {
        src_corr.push_back(p);
        dst_corr.push_back(best_q);
      }
    }

    // Only required if you want to reject outliers based on distance
    if (src_corr.size() < 5)
      break;

    // --- Compute centroids ---
    Eigen::Vector2d mu_src(0, 0), mu_dst(0, 0);
    for (size_t i = 0; i < src_corr.size(); i++)
    {
      mu_src += src_corr[i];
      mu_dst += dst_corr[i];
    }
    mu_src /= src_corr.size();
    mu_dst /= dst_corr.size();

    // --- Compute covariance ---
    Eigen::Matrix2d W = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < src_corr.size(); i++)
      W += (src_corr[i] - mu_src) * (dst_corr[i] - mu_dst).transpose();

    // --- SVD ---
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();

    Eigen::Matrix2d R_iter = V * U.transpose();

    // Ensure proper rotation
    if (R_iter.determinant() < 0)
    {
      V.col(1) *= -1;
      R_iter = V * U.transpose();
    }

    Eigen::Vector2d T_iter = mu_dst - R_iter * mu_src;

    // --- Apply transform ---
    for (auto& p : src_transformed)
    {
      p = R_iter * p + T_iter;
    }

    // --- Accumulate transform ---
    R = R_iter * R;
    T = R_iter * T + T_iter;
  }

  // --- Extract result ---
  dtheta = atan2(R(1,0), R(0,0));
  dx = T(0);
  dy = T(1);

  return true;
}

bool G2OBasedMapping::addScanMatchingEdge(int id1, int id2)
{
  auto v1 = graph_.vertex(id1);
  auto v2 = graph_.vertex(id2);

  if (!v1 || !v2) return false;

  // Get scans from user data
  g2o::RawLaser* raw1 = dynamic_cast<g2o::RawLaser*>(v1->userData());
  g2o::RawLaser* raw2 = dynamic_cast<g2o::RawLaser*>(v2->userData());

  if (!raw1 || !raw2) return false;

  // Keep points in their LOCAL frames
  auto pts1 = laserToPoints(raw1); 
  auto pts2 = laserToPoints(raw2);

  // 1. Extract the current global estimates to form a relative guess
  std::vector<double> p1, p2;
  v1->getEstimateData(p1);
  v2->getEstimateData(p2);

  g2o::SE2 pose1(p1[0], p1[1], p1[2]);
  g2o::SE2 pose2(p2[0], p2[1], p2[2]);

  // 2. Compute the relative transformation guess (Node 1 -> Node 2)
  g2o::SE2 relative_guess = pose1.inverse() * pose2;

  double dx_icp, dy_icp, dtheta_icp;

  // 3. Run ICP
  // Note: We align pts2 (source) to pts1 (target)
  if (computeICP(pts2, pts1, dx_icp, dy_icp, dtheta_icp, 
                 relative_guess.translation().x(), 
                 relative_guess.translation().y(), 
                 relative_guess.rotation().angle()))
  {
    // 4. Add the refined relative edge to the graph
    addLaserEdge(id1, id2, dx_icp, dy_icp, dtheta_icp, laser_noise_);
    return true;
  }

  return false;
}

// bool detectAndAddLoopClosures(int id)
// {
// #ifndef FIND_LOOP_CLOSURE
//   return false;
// #endif

//   bool loop_found = false;
  
//   // Get current pose
//   auto current_vertex = graph_.vertex(id);
//   if (!current_vertex) return false;

//   std::vector<double> pose_curr;
//   current_vertex->getEstimateData(pose_curr);

//   // 1. Collect all valid candidates
//   struct LoopCandidate {
//     int id;
//     double distance;
//   };
//   std::vector<LoopCandidate> candidates;

//   for (int candidate_id : robot_pose_ids_) 
//   {
//     // Skip recent nodes to avoid matching with the current trajectory
//     if (abs(id - candidate_id) < LOOP_CLOSURE_MIN_ID_DIFF) continue;

//     auto candidate_vertex = graph_.vertex(candidate_id);
//     if (!candidate_vertex) continue;

//     std::vector<double> pose_old;
//     candidate_vertex->getEstimateData(pose_old);

//     // Calculate distance between the global estimates
//     double dx = pose_curr[0] - pose_old[0];
//     double dy = pose_curr[1] - pose_old[1];
//     double dist = sqrt(dx * dx + dy * dy);

//     if (dist < LOOP_CLOSURE_SEARCH_RADIUS) 
//     {
//       candidates.push_back({candidate_id, dist});
//     }
//   }

//   // 2. Sort candidates by distance (closest first)
//   std::sort(candidates.begin(), candidates.end(), 
//     [](const LoopCandidate& a, const LoopCandidate& b) {
//       return a.distance < b.distance;
//   });

//   // 3. Try to connect to the 2 closest nodes
//   int closures_added = 0;
//   for (const auto& candidate : candidates) 
//   {
//     if (closures_added >= 2) break; // Limit to 2 closest candidates

//     // Try to add the edge using our robust relative ICP function
//     if (addScanMatchingEdge(candidate.id, id)) 
//     {
//       loop_found = true;
//       closures_added++;
//       RCLCPP_INFO(get_logger(), "Loop closure added! Node %d -> Node %d (Dist: %.2f)", 
//                   candidate.id, id, candidate.distance);
//     }
//   }

//   return loop_found;
// }

bool G2OBasedMapping::detectAndAddLoopClosures(int id)
{
#ifndef FIND_LOOP_CLOSURE
  return false;
#endif

  auto current_vertex = graph_.vertex(id);
  if (!current_vertex) return false;

  std::vector<double> p_curr;
  current_vertex->getEstimateData(p_curr);
  g2o::SE2 pose_curr(p_curr[0], p_curr[1], p_curr[2]);

  bool any_loop_added = false;
  int closures_this_cycle = 0;

  // 1. Candidate Selection
  // We iterate backwards to find the oldest matches first (better for global consistency)
  for (int i = 0; i < static_cast<int>(robot_pose_ids_.size()); ++i)
  {
    int candidate_id = robot_pose_ids_[i];

    // RULE 1: Temporal Exclusion 
    // Don't match with itself or very recent history (avoid the "tail")
    if (std::abs(id - candidate_id) < LOOP_CLOSURE_MIN_ID_DIFF) continue;

    auto candidate_vertex = graph_.vertex(candidate_id);
    if (!candidate_vertex) continue;

    std::vector<double> p_cand;
    candidate_vertex->getEstimateData(p_cand);
    
    double dist = std::sqrt(std::pow(p_curr[0] - p_cand[0], 2) + 
                            std::pow(p_curr[1] - p_cand[1], 2));

    // RULE 2: Spatial Proximity
    if (dist < LOOP_CLOSURE_SEARCH_RADIUS)
    {
      // 2. Attempt Scan Matching
      // Our addScanMatchingEdge handles the local ICP refinement
      if (addScanMatchingEdge(candidate_id, id))
      {        
        RCLCPP_INFO(get_logger(), "Loop Closure: Node %d <-> %d (Dist: %.2fm)", 
                    candidate_id, id, dist);
        any_loop_added = true;
        closures_this_cycle++;
      }
    }

    if (closures_this_cycle >= MAX_CLOSURES_PER_NODE) break;
  }

  return any_loop_added;
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::updateLaser(const LaserScan::ConstSharedPtr& laser)
{
  if (!laser_params_ || graph_.vertices().size() == 0)
  {
      // first laser update
      laser_params_ = new g2o::LaserParameters(
        laser->ranges.size(),
        laser->angle_min,
        laser->angle_increment,
        laser->range_max
      );

      RCLCPP_INFO_STREAM(get_logger(),
        "Adding first laser vertex " << last_id_ << " at pose: " << x_.transpose());
      addLaserVertex(x_(0), x_(1), x_(2), *laser, last_id_, true);
      return;
  }

  // Adding vertex if the robot has moved enough since the last vertex
  static Eigen::Vector3d last_vertex_pose = x_;

  double dx = x_(0) - last_vertex_pose(0);
  double dy = x_(1) - last_vertex_pose(1);
  double dtheta = atan2(sin(x_(2) - last_vertex_pose(2)), cos(x_(2) - last_vertex_pose(2)));

  double dist = sqrt(dx * dx + dy * dy);
  if (dist < TRANS_THRESHOLD && dtheta < ROT_THRESHOLD)
  {
    // Not enough motion → skip
    updateLocalization();
    visualizeRobotPoses();
    visualizeLaserScans();
    return;
  }

  int new_id = last_id_ + 1;
  RCLCPP_INFO_STREAM(get_logger(),
      "Adding laser vertex " << new_id << " at pose: " << x_.transpose());
  addLaserVertex(x_(0), x_(1), x_(2), *laser, new_id, false);
  addOdomEdge(last_id_, new_id);
  addScanMatchingEdge(last_id_, new_id);
  bool loop_found = detectAndAddLoopClosures(new_id);
  static int counter = 0;
  counter++;
  if (loop_found || counter % min_to_optimize_ == 0)
  {
    optimizeGraph();
    counter = 0;
  }
  last_id_ = new_id;
  last_vertex_pose = x_;

  // Keep This - reports your update
  updateLocalization();
  visualizeRobotPoses();
  // Keep This - if you like to visualize your map (collected laser scans in the graph)
  visualizeLaserScans();
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::init(double x, double y, double theta)
{
  x_(0, 0) = x;
  x_(1, 0) = y;
  x_(2, 0) = theta;

  graph_.clear();
  edge_set_.clear();
  vertex_set_.clear();
  seen_landmarks_.clear();
  robot_pose_ids_.clear();
  robot_landmark_edge_ids_.clear();
  laser_edge_ids_.clear();
  
  valid_ = false;
  reset_ = true;
  robot_pose_set_ = true;
  first_opt_ = true;

  min_to_optimize_ = MIN_TO_OPTIMIZE;
  last_id_ = 30;

  visualizeOldLandmarks();
  visualizeLandmarks();
  visualizeRobotPoses();
  visualizeEdges();
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::addOdomVertex(
  double x,
  double y,
  double theta,
  int id,
  bool first
)
{
  g2o::SE2 pose(x, y, theta);
  g2o::VertexSE2* vertex = new g2o::VertexSE2;
  vertex->setId(id);
  vertex->setEstimate(pose);

  graph_.addVertex(vertex);
  vertex_set_.insert(vertex);
  robot_pose_ids_.push_back(id);

  if(first)
    vertex->setFixed(true);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::addLaserVertex(
  double x,
  double y,
  double theta,
  LaserScan scan,
  int id,
  bool first
)
{
  g2o::SE2 pose(x, y, theta);
  g2o::VertexSE2* vertex = new g2o::VertexSE2;
  vertex->setId(id);
  vertex->setEstimate(pose);

  g2o::RawLaser* rl = new g2o::RawLaser();
  rl->setLaserParams(*laser_params_);

  std::vector<double> r;
  std::vector<float>::iterator it = scan.ranges.begin();

  r.assign(it, scan.ranges.end());
  rl->setRanges(r);
  vertex->addUserData(rl);
  graph_.addVertex(vertex);
  vertex_set_.insert(vertex);
  robot_pose_ids_.push_back(id);

  if(first)
    vertex->setFixed(true);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::addLaserEdge(
  int id1,
  int id2,
  double x,
  double y,
  double yaw,
  Eigen::Matrix3d noise
)
{
  g2o::EdgeSE2* edge = new g2o::EdgeSE2;
  edge->vertices()[0] = graph_.vertex(id1);
  edge->vertices()[1] = graph_.vertex(id2);
  edge->setMeasurement(g2o::SE2(x,y,yaw));
  edge->setInformation(noise);

  laser_edge_ids_.push_back(std::pair<int, int>(id1, id2));

  graph_.addEdge(edge);
  edge_set_.insert(edge);

  RCLCPP_INFO_STREAM(get_logger(), "added laser edge: " << id1 << " - " << id2);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::addOdomEdge(int id1, int id2)
{
  std::vector<double> data1,data2;

  graph_.vertex(id1)->getEstimateData(data1);
  graph_.vertex(id2)->getEstimateData(data2);

  g2o::SE2 vertex1(data1[0], data1[1], data1[2]);
  g2o::SE2 vertex2(data2[0], data2[1], data2[2]);

  g2o::SE2 transform = vertex1.inverse() * vertex2;
  g2o::EdgeSE2* edge = new g2o::EdgeSE2;
  edge->vertices()[0] = graph_.vertex(id1);
  edge->vertices()[1] = graph_.vertex(id2);
  edge->setMeasurement(transform);
  edge->setInformation(odom_noise_);

  graph_.addEdge(edge);
  edge_set_.insert(edge);

  RCLCPP_INFO_STREAM(
    get_logger(),
    "added odometry edge: " << id1 << " - " << id2
  );
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::addLandmarkVertex(double x, double y, int id)
{
  if(graph_.vertex(id))
    return;

  Eigen::Vector2d pos(x, y);
  g2o::VertexPointXY *vertex = new g2o::VertexPointXY;
  vertex->setId(id);
  vertex->setEstimate(pos);

  seen_landmarks_.push_back(id);
  graph_.addVertex(vertex);
  vertex_set_.insert(vertex);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::addLandmarkEdge(int id1, int id2, double x, double y)
{
  std::vector<double> data;
  graph_.vertex(id1)->getEstimateData(data);

  g2o::SE2 vertex1(data[0], data[1], data[2]);
  Eigen::Vector2d vertex2(x, y);
  Eigen::Vector2d measurement;
  measurement = vertex1.inverse() * vertex2;

  g2o::EdgeSE2PointXY* landmark_edge =  new g2o::EdgeSE2PointXY;
  landmark_edge->vertices()[0] = graph_.vertex(id1);
  landmark_edge->vertices()[1] = graph_.vertex(id2);
  landmark_edge->setMeasurement(measurement);
  landmark_edge->setInformation(landmark_noise_);

  graph_.addEdge(landmark_edge);
  edge_set_.insert(landmark_edge);
  robot_landmark_edge_ids_.push_back(std::pair<int, int>(id1, id2));

  RCLCPP_INFO_STREAM(
    get_logger(),
    "added landmark edge: " << id1 << " - " << id2
  );
}

// ----------------------------------------------------------------------------
void G2OBasedMapping::optimizeGraph()
{
  graph_.save("state_before.g2o");
  graph_.setVerbose(true);
  visualizeOldLandmarks();

  RCLCPP_INFO_STREAM(get_logger(), "Optimizing");

  if(first_opt_)
  {
    if(!graph_.initializeOptimization())
      RCLCPP_ERROR_STREAM(get_logger(), "FAILED initializeOptimization");
  }

  else if(!graph_.updateInitialization(vertex_set_, edge_set_))
    RCLCPP_ERROR_STREAM(get_logger(), "FAILED updateInitialization");

  graph_.optimize(OPTIMIZATION_ITERATIONS, !first_opt_);
  graph_.save("state_after.g2o");

  first_opt_ = false;
  vertex_set_.clear();
  edge_set_.clear();

  setRobotToVertex(robot_pose_ids_.back());
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::setRobotToVertex(int id)
{
  std::vector<double> data;
  graph_.vertex(id)->getEstimateData(data);

  x_(0, 0) = data[0];
  x_(1, 0) = data[1];
  x_(2, 0) = data[2];

  updateLocalization();
}

// -----------------------------------------------------------------------------
G2OBasedMapping::LaserScan G2OBasedMapping::rawLasertoLaserScanMsg(
  g2o::RawLaser rawlaser
)
{
  LaserScan msg;
  msg.header.frame_id = "base_laser_link";
  msg.angle_min = rawlaser.laserParams().firstBeamAngle;
  msg.angle_increment = rawlaser.laserParams().angularStep;
  msg.range_min = 0;
  msg.range_max = rawlaser.laserParams().maxRange;

  std::vector<double>::const_iterator it = rawlaser.ranges().begin();
  msg.ranges.assign(it, rawlaser.ranges().end());

  return msg;
}

// ----------------------------------------------------------------------------
void G2OBasedMapping::visualizeLaserScans()
{
  PointCloud graph_cloud;
  graph_cloud.header.frame_id = "map";
  graph_cloud.header.stamp = get_clock()->now();
  
  for(size_t j = 0; j < robot_pose_ids_.size(); j++)
  {
    std::vector<double> data;
    graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);

    g2o::OptimizableGraph::Data* d
      = graph_.vertex(robot_pose_ids_[j])->userData();

    g2o::RawLaser* rawLaser = dynamic_cast<g2o::RawLaser*>(d);

    if (rawLaser)
    {
      float angle = rawLaser->laserParams().firstBeamAngle;
      for(
        std::vector<double>::const_iterator i = rawLaser->ranges().begin();
        i != rawLaser->ranges().end();
        i++
      )
      {
        Point32 p;
        float x = *i * cos(angle);
        float y = *i * sin(angle);

        p.x = data[0] + x * cos(data[2]) - y * sin(data[2]);
        p.y = data[1] + x * sin(data[2]) + y * cos(data[2]);
        p.z = 0;
        angle += rawLaser->laserParams().angularStep;
        graph_cloud.points.push_back(p);
      }
    }
  }

  graph_cloud_publisher_->publish(graph_cloud);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::visualizeRobotPoses()
{
  Marker marker;
  MarkerArray marker_array;

  marker.header.frame_id = "map";
  marker.header.stamp = get_clock()->now();
  marker.ns = "robot_poses";
  marker.pose.position.z = 0.0;
  marker.type = Marker::SPHERE;
  marker.action = Marker::ADD;
  marker.scale.x = 0.2;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;
  marker.color.a = 0.5;
  marker.color.r = 0.1;
  marker.color.g = 0.1;
  marker.color.b = 0.9;

  for(size_t j = 0; j < robot_pose_ids_.size(); j++)
  {
    // Sphere Marker
    std::vector<double> data;
    graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);
    marker.pose.position.x = data[0];
    marker.pose.position.y = data[1];
    marker.id = robot_pose_ids_[j];
    marker_array.markers.push_back(marker);
  }

  robot_pose_marker_publisher_->publish(marker_array);

  visualizeEdges();
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::visualizeLandmarks()
{
  Marker marker;
  Marker marker_text;;
  MarkerArray marker_array;
  MarkerArray marker_array_text;

  marker.header.frame_id = "map";
  marker.header.stamp = get_clock()->now();
  marker.pose.position.z = 0.0;
  marker.ns = "observed_fiducials";
  marker.type = Marker::SPHERE;
  marker.action = Marker::ADD;
  marker.scale.x = 0.6;
  marker.scale.y = 0.6;
  marker.scale.z = 0.6;
  marker.color.a = 0.5;
  marker.color.r = 1.0;
  marker.color.g = 0.3;
  marker.color.b = 0.0;
  
  marker_text.header = marker.header;
  marker_text.ns = "observed_fiducials_text";
  marker_text.type = Marker::TEXT_VIEW_FACING;
  marker_text.action = Marker::ADD;
  marker_text.scale.z = 0.6 * 0.85;
  marker_text.color.a = 0.7;
  marker_text.color.r = 0.0;
  marker_text.color.g = 0.0;
  marker_text.color.b = 0.0;
  
  for(size_t j = 0; j < seen_landmarks_.size(); j++)
  {
    // Sphere Marker
    std::vector<double> data;
    graph_.vertex(seen_landmarks_[j])->getEstimateData(data);

    marker.pose.position.x = data[0];
    marker.pose.position.y = data[1];
    marker.id = seen_landmarks_[j];
    marker_array.markers.push_back(marker);

    // Text Marker
    marker_text.pose.position = marker.pose.position;
    marker_text.id = seen_landmarks_[j];
    marker_text.text = marker_text.id;
    marker_array_text.markers.push_back(marker_text);
  }

  fiducials_observed_marker_publisher_->publish(marker_array);
  fiducials_observed_marker_publisher_->publish(marker_array_text);

  visualizeEdges();
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::visualizeEdges()
{
  Marker marker;
  MarkerArray marker_array;

  marker.header.frame_id = "map";
  marker.header.stamp = get_clock()->now();
  marker.scale.x = 0.05;
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;
  marker.color.a = 0.5;
  marker.color.r = 0.9;
  marker.color.g = 0.1;
  marker.color.b = 0.1;
  marker.id = 0;
  marker.type = Marker::LINE_STRIP;
  marker.action = Marker::ADD;
  marker.ns = "edges";

  Point p;
  p.z = 0;

  std::vector<double> data;

  for(size_t j = 0; j < robot_pose_ids_.size(); j++)
  {
    graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);

    p.x = data[0];
    p.y = data[1];
    marker.points.push_back(p);
  }

  graph_edges_publisher_->publish(marker);

  marker.points.clear();
  marker.id = 1;
  marker.type = Marker::LINE_LIST;

  for(size_t j = 0; j < robot_landmark_edge_ids_.size(); j++)
  {
    graph_.vertex(robot_landmark_edge_ids_[j].first)->getEstimateData(data);
    p.x = data[0];
    p.y = data[1];
    marker.points.push_back(p);

    graph_.vertex(robot_landmark_edge_ids_[j].second)->getEstimateData(data);
    p.x = data[0];
    p.y = data[1];
    marker.points.push_back(p);
  }

  for(size_t j = 0; j < laser_edge_ids_.size(); j++)
  {
    graph_.vertex(laser_edge_ids_[j].first)->getEstimateData(data);
    p.x = data[0];
    p.y = data[1];
    marker.points.push_back(p);

    graph_.vertex(laser_edge_ids_[j].second)->getEstimateData(data);
    p.x = data[0];
    p.y = data[1];
    marker.points.push_back(p);
  }

  graph_edges_publisher_->publish(marker);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::visualizeOldLandmarks()
{
  Marker marker;
  Marker marker_text;;
  MarkerArray marker_array;
  MarkerArray marker_array_text;

  marker.header.frame_id = "map";
  marker.header.stamp = get_clock()->now();
  marker.pose.position.z = 0.0;
  marker.type = Marker::SPHERE;
  marker.action = Marker::ADD;
  marker.scale.x = 0.6;
  marker.scale.y = 0.6;
  marker.scale.z = 0.6;
  marker.color.a = 0.5;
  marker.color.r = 0.1;
  marker.color.g = 0.1;
  marker.color.b = 0.8;
  marker.ns = "old_observed_fiducials";

  marker_text.header = marker.header;
  marker_text.pose.position = marker.pose.position;
  marker_text.type = Marker::TEXT_VIEW_FACING;
  marker_text.action = Marker::ADD;
  marker_text.scale.z = 0.6*0.85;
  marker_text.color.a = 0.7;
  marker_text.color.r = 0.0;
  marker_text.color.g = 0.0;
  marker_text.color.b = 0.0;
  marker_text.ns = "old_observed_fiducials_text";

  for(size_t j = 0; j < seen_landmarks_.size(); j++)
  {
    // Sphere Marker
    std::vector<double> data;
    graph_.vertex(seen_landmarks_[j])->getEstimateData(data);

    marker.pose.position.x = data[0];
    marker.pose.position.y = data[1];
    marker.id = seen_landmarks_[j];
    marker_array.markers.push_back(marker);

    // Text Marker
    marker_text.id = seen_landmarks_[j];
    marker_text.text = marker_text.id;
    marker_array_text.markers.push_back(marker_text);
  }

  old_fiducials_observed_marker_publisher_->publish(marker_array);
  old_fiducials_observed_marker_publisher_->publish(marker_array_text);
}

// ----------------------------------------------------------------------------
void G2OBasedMapping::updateLocalization()
{
  TransformStamped transform;
  transform.header.frame_id = "map";
  transform.header.stamp = get_clock()->now();
  transform.child_frame_id = "base_link_g2o";
  transform.transform.translation.x = x_(0);
  transform.transform.translation.y = x_(1);

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, x_(2));

  transform.transform.rotation = tf2::toMsg(q);

  transform_broadcaster_->sendTransform(transform);
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::publishMap()
{
  data_mutex_.lock();
  
  graph_map_.header.stamp = get_clock()->now();
  graph_map_.info.map_load_time = get_clock()->now();

  int map_size = graph_map_.info.width * graph_map_.info.height;
  graph_map_.data = std::vector<int8_t>(map_size, 0);

  for(size_t j = 0; j < robot_pose_ids_.size(); j++)
  {
    std::vector<double> data;
    graph_.vertex(robot_pose_ids_[j])->getEstimateData(data);

    g2o::OptimizableGraph::Data* d
      = graph_.vertex(robot_pose_ids_[j])->userData();

    g2o::RawLaser* rawLaser = dynamic_cast<g2o::RawLaser*>(d);

    if (rawLaser)
    {
      float angle = rawLaser->laserParams().firstBeamAngle;

      for(
        std::vector<double>::const_iterator i = rawLaser->ranges().begin();
        i != rawLaser->ranges().end();
        i++
      )
      {
        Point32 p;
        float x = *i * cos(angle);
        float y = *i * sin(angle);

        p.x = data[0] + x * cos(data[2]) - y * sin(data[2])
          - graph_map_.info.origin.position.x;

        p.y = data[1] + x * sin(data[2]) + y * cos(data[2])
          - graph_map_.info.origin.position.y;

        angle += rawLaser->laserParams().angularStep;

        unsigned int map_x = p.x / graph_map_.info.resolution;
        unsigned int map_y = p.y / graph_map_.info.resolution;

        if (
          map_x < graph_map_.info.width &&
          map_y < graph_map_.info.height
        )
        {
          graph_map_.data[map_y * graph_map_.info.width + map_x] = (int8_t) 100;
        }
      }
    }
  }

  map_publisher_->publish(graph_map_);

  data_mutex_.unlock();
}

// -----------------------------------------------------------------------------
double G2OBasedMapping::yawFromQuaternion(const Quaternion& quaternion)
{
  double roll;
  double pitch;
  double yaw;

  tf2::Quaternion q;
  tf2::fromMsg(quaternion, q);

  tf2::Matrix3x3 m(q);
  m.getRPY(roll, pitch, yaw);

  return yaw;
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::odomCallback(const Odometry::ConstSharedPtr& msg)
{
  data_mutex_.lock();
  updateOdometry(msg);
  data_mutex_.unlock();
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::laserCallback(const LaserScan::ConstSharedPtr& msg)
{
  data_mutex_.lock();
  updateLaser(msg);
  data_mutex_.unlock();
}

// -----------------------------------------------------------------------------
void G2OBasedMapping::initialPoseCallback(
  const PoseWithCovarianceStamped::ConstSharedPtr& msg
)
{
  data_mutex_.lock();

  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;
  double yaw = yawFromQuaternion(msg->pose.pose.orientation);

  RCLCPP_INFO_STREAM(
    get_logger(),
    "initialPoseCallback x: " << x << ", y: " << y << ", theta: " << yaw
  ); 

  init(x, y, yaw);

  data_mutex_.unlock();
}

} /* namespace tug_g2o_based_mapping */
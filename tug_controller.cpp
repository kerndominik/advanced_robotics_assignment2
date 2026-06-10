#include "tug_controller/tug_controller.hpp"
#include "pluginlib/class_list_macros.hpp"

#define LOGGING true

#if LOGGING 
  #include <fstream>
  #include <iomanip>
  std::ofstream log_file;
  rclcpp::Time start_time;
  bool is_first_log = true;
#endif

// Robot parameters
#define FRONT_AXLE  0.256
#define FRONT_BUMPER 0.481
#define WHEELBASE   0.512
#define TRACK_WIDTH 0.555
#define MAX_STEERING_ANGLE  3.1415 / 3

// Controller parameters
#define K_STANLEY   0.25
#define K_S         0.26
// PI parameters found with "T-Summenregel"
#define K_P         0.5
#define K_I         4.0
#define ACCEL_LIM   0.3

// Replanning thresholds
#define GOAL_UPDATE_THRESHOLD 0.01

// Lookahead for closest point search
#define INDICES_LOOKAHEAD 5

// Path smoothing parameters
#define PATH_SMOOTHING true
#define SMOOTHING_ALPHA 0.5
#define SMOOTHING_ITERATIONS 3

// collision detection
#define COST_LETHAL_OBSTACLE 252

namespace tug_controller
{

// -----------------------------------------------------------------------------
void TugController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& node,
  std::string controller_name,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cost_map
)
{
  node_ = node.lock();
  controller_name_ = controller_name;
  tf_buffer_ = tf_buffer;
  cost_map_ = cost_map;
  smooth_path_pub_ =
  node_->create_publisher<nav_msgs::msg::Path>(
    "/plan_smoothed",
    1);

#if LOGGING
  std::string file_path = "/tmp/tug_controller_telemetry.csv";
  log_file.open(file_path, std::ios::out);
  
  if (log_file.is_open()) {
    // Write CSV Header
    log_file << "timestamp_s,heading_error_rad,crosstrack_error_m,measured_velocity_x_mps\n";
    RCLCPP_INFO(node_->get_logger(), "Successfully opened CSV log file at: %s", file_path.c_str());
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Failed to open CSV log file at: %s", file_path.c_str());
  }
#endif
}

// -----------------------------------------------------------------------------
void TugController::activate()
{
  RCLCPP_INFO(node_->get_logger(), "Activating tug controller");
}

nav_msgs::msg::Path TugController::smoothPath(const Path& path)
{
  if(path.poses.size() < 3)
    return path;

  Path smoothed = path;

  const double alpha = SMOOTHING_ALPHA;   // smoothing strength
  const int iterations = SMOOTHING_ITERATIONS;

  for(int iter = 0; iter < iterations; iter++)
  {
    Path temp = smoothed;

    for(size_t i = 1; i < path.poses.size() - 1; i++)
    {
      double x_prev = temp.poses[i-1].pose.position.x;
      double x_curr = temp.poses[i].pose.position.x;
      double x_next = temp.poses[i+1].pose.position.x;

      double y_prev = temp.poses[i-1].pose.position.y;
      double y_curr = temp.poses[i].pose.position.y;
      double y_next = temp.poses[i+1].pose.position.y;

      double x_avg = (x_prev + x_curr + x_next) / 3.0;
      double y_avg = (y_prev + y_curr + y_next) / 3.0;

      smoothed.poses[i].pose.position.x =
          x_curr + alpha * (x_avg - x_curr);

      smoothed.poses[i].pose.position.y =
          y_curr + alpha * (y_avg - y_curr);
    }
  }

  // Recompute orientations
  for(size_t i = 0; i < smoothed.poses.size()-1; i++)
  {
    double dx =
        smoothed.poses[i+1].pose.position.x -
        smoothed.poses[i].pose.position.x;

    double dy =
        smoothed.poses[i+1].pose.position.y -
        smoothed.poses[i].pose.position.y;

    double yaw = atan2(dy, dx);

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);

    smoothed.poses[i].pose.orientation =
        tf2::toMsg(q);
  }

  // Last pose gets same orientation as previous
  smoothed.poses.back().pose.orientation =
      smoothed.poses[smoothed.poses.size()-2].pose.orientation;

  return smoothed;
}

// -----------------------------------------------------------------------------
void TugController::setPlan(const Path& path)
{
  if (path.poses.empty())
  {
    RCLCPP_WARN(node_->get_logger(), "Received empty path");
    return;
  }
  static bool first_call = true;
  if (first_call)
  {
    RCLCPP_INFO(node_->get_logger(), "Received new path with %zu poses", path.poses.size());
    global_path_ = PATH_SMOOTHING ? smoothPath(path) : path;
    smooth_path_pub_->publish(global_path_);
    last_path_received_time_ = node_->now();
    first_call = false;
    setSpeedLimit(100.0, true);
    last_longitudinal_control_time_ = node_->now();
    return;
  }
  const auto& new_goal = path.poses.back().pose;
  const auto& old_goal = global_path_.poses.back().pose;
  double dx = new_goal.position.x - old_goal.position.x;
  double dy = new_goal.position.y - old_goal.position.y;
  double goal_dist = std::hypot(dx, dy);
  
  if (goal_dist < GOAL_UPDATE_THRESHOLD) return;

  RCLCPP_INFO(node_->get_logger(), "Received new path with %zu poses", path.poses.size());
  global_path_ = PATH_SMOOTHING ? smoothPath(path) : path;
  smooth_path_pub_->publish(global_path_);
  last_path_received_time_ = node_->now();
  setSpeedLimit(100.0, true);
  last_longitudinal_control_time_ = node_->now();
  integral_error_ = 0.0;
}

double TugController::lateralControl(const PoseStamped& pose, const Twist& velocity){
  // Robot state
  double x = pose.pose.position.x;
  double y = pose.pose.position.y;
  double yaw = tf2::getYaw(pose.pose.orientation);

  // Front axle position
  double fx = x + FRONT_AXLE * cos(yaw);
  double fy = y + FRONT_AXLE * sin(yaw);

  // Find closest path point
  int closest_idx = global_path_.poses.size() - 1;
  double min_dist = std::numeric_limits<double>::max();
  for(size_t i = 0; i < global_path_.poses.size() - INDICES_LOOKAHEAD; i++)
  {
    double px = global_path_.poses[i].pose.position.x;
    double py = global_path_.poses[i].pose.position.y;
    double d = hypot(fx-px, fy-py);
    if(d < min_dist)
    {
      min_dist = d;
      closest_idx = i + INDICES_LOOKAHEAD;
    }
  }

  auto target = global_path_.poses[closest_idx];
  double tx = target.pose.position.x;
  double ty = target.pose.position.y;
  double target_yaw = tf2::getYaw(target.pose.orientation);

  // Heading error
  double heading_error = atan2(sin(target_yaw - yaw), cos(target_yaw - yaw));

  // Crosstrack error
  double dx = tx - fx;
  double dy = ty - fy;
  cross_track_error_ = -sin(target_yaw)*dx + cos(target_yaw)*dy;

#if LOGGING
  if (log_file.is_open()) {
    rclcpp::Time current_time = node_->now();
    if (is_first_log) {
      start_time = current_time;
      is_first_log = false;
    }
    double elapsed_time = (current_time - start_time).seconds();

    log_file << std::fixed << std::setprecision(4)
             << elapsed_time << ","
             << heading_error << ","
             << cross_track_error_ << ","
             << velocity.linear.x << "\n";
  }
#endif

  // Stanley control
  double atan_value = atan2(K_STANLEY*cross_track_error_,K_S + velocity.linear.x);
  double delta = heading_error + atan_value;
  delta = std::clamp(delta, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE);
  double omega = velocity.linear.x / WHEELBASE * tan(delta);
  // RCLCPP_INFO(node_->get_logger(), "heading error: %.2f; crosstrack error: %.2f; atan value: %.2f; delta: %.2f; omega: %.2f", heading_error, cross_track_error_, atan_value, delta, omega);
  return omega;
}

double TugController::longitudinalControl(const Twist& velocity)
{
  rclcpp::Time current_time = node_->now();
  double dt = (current_time - last_longitudinal_control_time_).seconds();
  last_longitudinal_control_time_ = current_time;

  double speed_error = v_ref_ - velocity.linear.x;
  double unsat_output = K_P * speed_error + K_I * integral_error_;

  double sat_output = std::clamp(unsat_output, -max_speed_, max_speed_);

  // Conditional integration (anti-windup)
  bool controller_saturated = std::abs(unsat_output - sat_output) > 1e-6;
  bool error_reduces_saturation =
      (unsat_output > max_speed_ && speed_error < 0.0) ||
      (unsat_output < -max_speed_ && speed_error > 0.0);

  if(!controller_saturated || error_reduces_saturation)
    integral_error_ += speed_error * dt;
  
  // Recompute output using updated integral
  double v = K_P * speed_error + K_I * integral_error_;

  // Saturation again
  v = std::clamp(v, -max_speed_, max_speed_);

  // Acceleration limiting
  double dv = v - previous_cmd_vel_;
  double max_dv = ACCEL_LIM * dt;
  if(dv > max_dv)
    v = previous_cmd_vel_ + max_dv;
  if(dv < -max_dv)
    v = previous_cmd_vel_ - max_dv;
  previous_cmd_vel_ = v;
  return v;
}

// -----------------------------------------------------------------------------
TugController::TwistStamped TugController::computeVelocityCommands(
  const PoseStamped& pose,
  const Twist& velocity,
  nav2_core::GoalChecker* goal_checker
)
{
  TwistStamped cmd;
  cmd.header.stamp = node_->now();
  cmd.header.frame_id = "base_link";
  double omega = lateralControl(pose, velocity);  // always call lateralContol to log the data
  if(global_path_.poses.empty())
  {
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }
  if(goal_checker && global_path_.poses.size() > 0)
  {
    auto goal_pose = global_path_.poses.back();
    if(goal_checker->isGoalReached(pose.pose, goal_pose.pose, velocity))
    {
      RCLCPP_INFO(node_->get_logger(), "Goal reached! Stopping.");
      setSpeedLimit(0.0, false);
      cmd.twist.linear.x = 0.0;
      cmd.twist.angular.z = 0.0;
      return cmd;
    }
  }

  double v = longitudinalControl(velocity);

  // Output command
  cmd.twist.linear.x = v;
  cmd.twist.angular.z = omega;

  // TODO: implement collision detection
  // stop if the front bumper is too close to an obstacle using the costmap (stop if costmap value at front bumper position >252)
  // positive omega (+0.8) if the cost map value of the front right wheel is > 252 and the front left wheel is not
  // negative omega (-0.8) if the cost map value of the front left wheel is > 252 and the front right wheel is not
  // stop if the cost map values of both front wheels are > 252

  // --- COLLISION DETECTION IMPLEMENTATION ---
  if (cost_map_)
  {
    // 1. Get the underlying Costmap2D map pointer
    nav2_costmap_2d::Costmap2D* costmap = cost_map_->getCostmap();
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));

    // 2. Extract robot current global state
    double robot_x = pose.pose.position.x;
    double robot_y = pose.pose.position.y;
    double robot_yaw = tf2::getYaw(pose.pose.orientation);

    // Helper lambda to calculate world coordinates based on base_link offsets
    auto get_world_pt = [&](double offset_x, double offset_y, double& wx, double& wy) {
      wx = robot_x + offset_x * cos(robot_yaw) - offset_y * sin(robot_yaw);
      wy = robot_y + offset_x * sin(robot_yaw) + offset_y * cos(robot_yaw);
    };

    // Helper lambda to get cost from global coordinates
    auto get_cost_at_world = [&](double wx, double wy) -> unsigned char {
      unsigned int mx, my;
      if (costmap->worldToMap(wx, wy, mx, my)) {
        return costmap->getCost(mx, my);
      }
      return nav2_costmap_2d::NO_INFORMATION; // Out of map bounds defaults to unsafe
    };

    // 3. Define check points (Offsets from base_link)
    double bumper_wx, bumper_wy;
    get_world_pt(FRONT_BUMPER, 0.0, bumper_wx, bumper_wy);

    double fl_wheel_wx, fl_wheel_wy;
    get_world_pt(FRONT_AXLE, TRACK_WIDTH / 2.0, fl_wheel_wx, fl_wheel_wy);

    double fr_wheel_wx, fr_wheel_wy;
    get_world_pt(FRONT_AXLE, -TRACK_WIDTH / 2.0, fr_wheel_wx, fr_wheel_wy);

    // 4. Sample cost levels
    unsigned char bumper_cost   = get_cost_at_world(bumper_wx, bumper_wy);
    unsigned char fl_wheel_cost = get_cost_at_world(fl_wheel_wx, fl_wheel_wy);
    unsigned char fr_wheel_cost = get_cost_at_world(fr_wheel_wx, fr_wheel_wy);

    // 5. Evaluate Collision Risk Rules
    bool stop_needed = false;

    // Rule A: Stop if the front bumper is too close to an obstacle (> 252)
    if (bumper_cost >= COST_LETHAL_OBSTACLE) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "Collision Mitigation: Front Bumper close to obstacle!");
      stop_needed = true;
    }
    // Rule B: Stop if both front wheels hit obstacle (> 252)
    else if (fl_wheel_cost >= COST_LETHAL_OBSTACLE && fr_wheel_cost >= COST_LETHAL_OBSTACLE) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "Collision Mitigation: Both front wheels blocked!");
      stop_needed = true;
    }
    
    if (stop_needed) {
      v = 0.0;
      omega = 0.0;
    } 
    else {
      // Rule C: Positive omega (+0.8) if right wheel hits obstacle and left does not
      if (fr_wheel_cost >= COST_LETHAL_OBSTACLE && fl_wheel_cost < COST_LETHAL_OBSTACLE) {
        omega = v / WHEELBASE * tan(MAX_STEERING_ANGLE); // steer left;
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "Collision Mitigation: Front right wheel close to obstacle, steering left!");
      }
      // Rule D: Negative omega (-0.8) if left wheel hits obstacle and right does not
      else if (fl_wheel_cost >= COST_LETHAL_OBSTACLE && fr_wheel_cost < COST_LETHAL_OBSTACLE) {
        omega = -v / WHEELBASE * tan(MAX_STEERING_ANGLE); // steer right;
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "Collision Mitigation: Front left wheel close to obstacle, steering right!");
      }
    }
  }

  return cmd;
}

// -----------------------------------------------------------------------------
void TugController::setSpeedLimit(
  const double& speed_limit,
  const bool& percentage
)
{
  // RCLCPP_INFO(node_->get_logger(), "Setting speed limit for tug controller");
  v_ref_ = percentage ? max_speed_ * speed_limit / 100.0 : speed_limit;
  std::clamp(v_ref_, 0.0, max_speed_);
}

// -----------------------------------------------------------------------------
void TugController::deactivate()
{
  RCLCPP_INFO(node_->get_logger(), "Deactivating tug controller");
}

// -----------------------------------------------------------------------------
void TugController::cleanup()
{
  RCLCPP_INFO(node_->get_logger(), "Cleaning up tug controller");
#if LOGGING
  if (log_file.is_open()) {
    log_file.close();
  }
#endif
}

} /* namespace tug_controller */

PLUGINLIB_EXPORT_CLASS(tug_controller::TugController, nav2_core::Controller);
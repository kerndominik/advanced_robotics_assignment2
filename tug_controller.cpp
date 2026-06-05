#include "tug_controller/tug_controller.hpp"

#include "pluginlib/class_list_macros.hpp"

#define FRONT_AXLE  0.256
#define WHEELBASE   0.512
#define K_STANLEY   4.0
#define K_S         0.5
#define K_P         3.0
#define K_I         0.0
#define ACCEL_LIM   0.3
#define PI          3.1415

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
  last_longitudinal_control_time_ = node_->now();
}

// -----------------------------------------------------------------------------
void TugController::activate()
{
  RCLCPP_INFO(node_->get_logger(), "Activating tug controller");
}

Path TugController::smoothPath(const Path& path)
{
    if(path.poses.size() < 3)
        return path;

    Path smoothed = path;

    const double alpha = 0.25;   // smoothing strength
    const int iterations = 3;

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
  static bool first_call = true;
  if (first_call)
  {
    RCLCPP_INFO(node_->get_logger(), "Received new path with %zu poses", path.poses.size());
    global_path_ = path;
    first_call = false;
    last_path_received_time_ = node_->now();
    return;
  }

  rclcpp::Time current_time = node_->now();
  
  if ((current_time - last_path_received_time_).seconds() < 3) return;

  last_path_received_time_ = current_time;
  RCLCPP_INFO(node_->get_logger(), "Received new path with %zu poses", path.poses.size());
  global_path_ = smoothPath(path);
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
  int closest_idx = 0;
  double min_dist = std::numeric_limits<double>::max();
  for(size_t i = 0; i < global_path_.poses.size(); i++)
  {
    double px = global_path_.poses[i].pose.position.x;
    double py = global_path_.poses[i].pose.position.y;
    double d = hypot(fx-px, fy-py);
    if(d < min_dist)
    {
      min_dist = d;
      closest_idx = i;
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
  double cross_track_error = -sin(target_yaw)*dx + cos(target_yaw)*dy;

  // Stanley control
  double atan_value = atan2(K_STANLEY*cross_track_error,K_S + velocity.linear.x);
  double delta = heading_error + atan_value;
  delta = std::clamp(delta, -PI/3, PI/3);
  double omega = velocity.linear.x / WHEELBASE * tan(delta);
  RCLCPP_INFO(node_->get_logger(), "heading error: %.2f; crosstrack error: %.2f; atan value: %.2f; delta: %.2f; omega: %.2f", heading_error, cross_track_error, atan_value, delta, omega);
  return omega;
}

double TugController::longitudinalControl(const Twist& velocity)
{
    rclcpp::Time current_time = node_->now();
    double dt = (current_time - last_longitudinal_control_time_).seconds();
    last_longitudinal_control_time_ = current_time;
  
    double speed_error = max_speed_ - velocity.linear.x;
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
      cmd.twist.linear.x = 0.0;
      cmd.twist.angular.z = 0.0;
      return cmd;
    }
  }

  double omega = lateralControl(pose, velocity);
  double v = longitudinalControl(velocity);

  // Output command
  cmd.twist.linear.x = v;
  cmd.twist.angular.z = omega;

  return cmd;
}

// -----------------------------------------------------------------------------
void TugController::setSpeedLimit(
  const double& speed_limit,
  const bool& percentage
)
{
  // RCLCPP_INFO(node_->get_logger(), "Setting speed limit for tug controller");
  max_speed_ = percentage ? max_speed_ * speed_limit / 100.0 : speed_limit;
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
}

} /* namespace tug_controller */

PLUGINLIB_EXPORT_CLASS(tug_controller::TugController, nav2_core::Controller);

#include "vnoid_challenge/balance_controller.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace vnoid_challenge
{
BalanceController::BalanceController(const rclcpp::NodeOptions& options)
    : BalanceController("balance_controller", "", options)
{
}

BalanceController::BalanceController(const std::string& node_name,
                                     const std::string& namespace_,
                                     const rclcpp::NodeOptions& options)
    : Node(node_name, namespace_, options)
{
  wheel_radius_ = declare_parameter<double>("wheel_radius");
  wheel_tread_ = declare_parameter<double>("wheel_tread");
  wheel_inertia_ = declare_parameter<double>("wheel_inertia");
  feedback_gain_ = declare_parameter<std::vector<double>>("feedback_gain");
  control_rate_ = declare_parameter<double>("control_rate");

  parameter_event_handler_
      = std::make_shared<rclcpp::ParameterEventHandler>(this);
  parameter_callback_handles_.push_back(
      parameter_event_handler_->add_parameter_callback(
          "feedback_gain", [this](const rclcpp::Parameter& p)
          { feedback_gain_ = p.as_double_array(); }));

  wheel_torque_command_left_pub_ = create_publisher<std_msgs::msg::Float32>(
      "wheel_torque_command/left", 1);
  wheel_torque_command_right_pub_ = create_publisher<std_msgs::msg::Float32>(
      "wheel_torque_command/right", 1);

  joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::SensorDataQoS(),
      std::bind(&BalanceController::jointStatesCallback, this,
                std::placeholders::_1));
  cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 1,
      std::bind(&BalanceController::cmdVelCallback, this,
                std::placeholders::_1));

  control_timer_
      = create_wall_timer(rclcpp::Rate(control_rate_).period(),
                          std::bind(&BalanceController::control, this));
}

void BalanceController::jointStatesCallback(
    const sensor_msgs::msg::JointState::ConstSharedPtr msg)
{
  for (size_t i = 0; i < msg->name.size(); ++i)
  {
    if (msg->name[i] == "BODY")
    {
      body_pitch_angle_ = msg->position[i];
      body_pitch_rate_ = msg->velocity[i];
    }
    else if (msg->name[i] == "L_WHEEL")
    {
      left_wheel_velocity_ = msg->velocity[i];
    }
    else if (msg->name[i] == "R_WHEEL")
    {
      right_wheel_velocity_ = msg->velocity[i];
    }
  }
}

void BalanceController::cmdVelCallback(
    const geometry_msgs::msg::Twist::ConstSharedPtr msg)
{
  cmd_vel_ = *msg;
}

void BalanceController::control()
{
  const auto balance_accel_command = -(feedback_gain_[0] * body_pitch_angle_
                                       + feedback_gain_[1] * body_pitch_rate_);

  const auto left_wheel_velocity_command
      = -(cmd_vel_.linear.x + wheel_tread_ / 2 * cmd_vel_.angular.z)
        / (2 * wheel_radius_);
  const auto right_wheel_velocity_command
      = -(cmd_vel_.linear.x - wheel_tread_ / 2 * cmd_vel_.angular.z)
        / (2 * wheel_radius_);

  const auto left_wheel_torque_command
      = balance_accel_command / (2 * wheel_radius_)
        + wheel_inertia_ * (left_wheel_velocity_command - left_wheel_velocity_)
              * control_rate_;
  const auto right_wheel_torque_command
      = balance_accel_command / (2 * wheel_radius_)
        + wheel_inertia_
              * (right_wheel_velocity_command - right_wheel_velocity_)
              * control_rate_;

  std_msgs::msg::Float32 left_wheel_torque_command_msg;
  left_wheel_torque_command_msg.data = left_wheel_torque_command;
  wheel_torque_command_left_pub_->publish(left_wheel_torque_command_msg);

  std_msgs::msg::Float32 right_wheel_torque_command_msg;
  right_wheel_torque_command_msg.data = right_wheel_torque_command;
  wheel_torque_command_right_pub_->publish(right_wheel_torque_command_msg);
}
}  // namespace vnoid_challenge

RCLCPP_COMPONENTS_REGISTER_NODE(vnoid_challenge::BalanceController)

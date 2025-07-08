#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>

namespace vnoid_challenge
{
class BalanceController : public rclcpp::Node
{
 public:
  explicit BalanceController(const rclcpp::NodeOptions& options
                             = rclcpp::NodeOptions());

  explicit BalanceController(const std::string& node_name,
                             const std::string& namespace_,
                             const rclcpp::NodeOptions& options
                             = rclcpp::NodeOptions());

 private:
  void jointStatesCallback(
      const sensor_msgs::msg::JointState::ConstSharedPtr msg);
  void cmdVelCallback(const geometry_msgs::msg::Twist::ConstSharedPtr msg);

  void control();

  double wheel_radius_;
  double wheel_tread_;
  double wheel_inertia_;
  std::vector<double> feedback_gain_;
  double control_rate_;

  double body_pitch_angle_;
  double body_pitch_rate_;
  double left_wheel_velocity_;
  double right_wheel_velocity_;
  geometry_msgs::msg::Twist cmd_vel_;

  std::shared_ptr<rclcpp::ParameterEventHandler> parameter_event_handler_;
  std::vector<rclcpp::ParameterCallbackHandle::SharedPtr>
      parameter_callback_handles_;

  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr
      wheel_torque_command_left_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr
      wheel_torque_command_right_pub_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      joint_states_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

  rclcpp::TimerBase::SharedPtr control_timer_;
};
}  // namespace vnoid_challenge

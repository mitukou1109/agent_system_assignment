#include <cnoid/Imu>
#include <cnoid/SimpleController>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>

namespace vnoid_challenge
{
class InvertedPendulumController : public cnoid::SimpleController
{
 public:
  bool configure(cnoid::SimpleControllerConfig* config) override
  {
    node_ = std::make_shared<rclcpp::Node>(config->controllerName());
    imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>(
        "imu/data", rclcpp::SensorDataQoS());
    joint_states_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::SensorDataQoS());
    wheel_torque_command_subs_.left()
        = node_->create_subscription<std_msgs::msg::Float32>(
            "wheel_torque_command/left", 1,
            [this](const std_msgs::msg::Float32::ConstSharedPtr msg)
            { wheel_torque_command_.left() = msg->data; });
    wheel_torque_command_subs_.right()
        = node_->create_subscription<std_msgs::msg::Float32>(
            "wheel_torque_command/right", 1,
            [this](const std_msgs::msg::Float32::ConstSharedPtr msg)
            { wheel_torque_command_.right() = msg->data; });

    executor_
        = std::make_unique<rclcpp::executors::StaticSingleThreadedExecutor>();
    executor_->add_node(node_);

    return true;
  }

  bool initialize(cnoid::SimpleControllerIO* io) override
  {
    for (auto& wheel : wheels_)
    {
      wheel.link = io->body()->link(wheel.name);
      wheel.link->setActuationMode(cnoid::Link::JointTorque);
      io->enableOutput(wheel.link);
      io->enableInput(wheel.link,
                      cnoid::Link::JointAngle | cnoid::Link::JointVelocity);
    }

    imu_ = io->body()->findDevice<cnoid::Imu>("IMU");
    io->enableInput(imu_);

    control_period_ = io->timeStep();

    body_pitch_angle_ = 0;
    body_pitch_rate_ = 0;

    wheel_torque_command_.left() = 0;
    wheel_torque_command_.right() = 0;

    wheels_.left().link->u() = 0;
    wheels_.right().link->u() = 0;

    return true;
  }

  bool control() override
  {
    executor_->spin_some();

    body_pitch_rate_ = imu_->w().y();
    body_pitch_angle_ += body_pitch_rate_ * control_period_;

    sensor_msgs::msg::Imu imu_msg;
    imu_msg.header.stamp = node_->now();
    imu_msg.header.frame_id = imu_->link()->name();
    imu_msg.linear_acceleration.x = imu_->dv().x();
    imu_msg.linear_acceleration.y = imu_->dv().y();
    imu_msg.linear_acceleration.z = imu_->dv().z();
    imu_msg.angular_velocity.x = imu_->w().x();
    imu_msg.angular_velocity.y = imu_->w().y();
    imu_msg.angular_velocity.z = imu_->w().z();
    imu_msg.orientation_covariance[0] = -1;
    imu_pub_->publish(imu_msg);

    sensor_msgs::msg::JointState joint_states_msg;
    joint_states_msg.header.stamp = node_->now();
    joint_states_msg.header.frame_id = "BASE_LINK";
    joint_states_msg.name = {"BODY", "L_WHEEL", "R_WHEEL"};
    joint_states_msg.position = {body_pitch_angle_, wheels_.left().link->q(),
                                 wheels_.right().link->q()};
    joint_states_msg.velocity = {body_pitch_rate_, wheels_.left().link->dq(),
                                 wheels_.right().link->dq()};
    joint_states_pub_->publish(joint_states_msg);

    wheels_.left().link->u()
        = std::clamp(wheel_torque_command_.left(), -1.0f, 1.0f);
    wheels_.right().link->u()
        = std::clamp(wheel_torque_command_.right(), -1.0f, 1.0f);

    return true;
  }

  void unconfigure() override
  {
    if (!executor_)
    {
      return;
    }

    executor_->cancel();
    executor_->remove_node(node_);
    executor_.reset();
  }

 private:
  struct Link
  {
    cnoid::LinkPtr link;
    std::string name;
  };

  template <typename T>
  class WheelSet
  {
   public:
    WheelSet() {}
    WheelSet(const T& left, const T& right) : values_{left, right} {}

    T& left() { return values_[0]; }
    T& right() { return values_[1]; }

    const T& left() const { return values_[0]; }
    const T& right() const { return values_[1]; }

    typename std::array<T, 2>::iterator begin() { return values_.begin(); }
    typename std::array<T, 2>::iterator end() { return values_.end(); }

   private:
    std::array<T, 2> values_;
  };

  WheelSet<Link> wheels_{{.name = "L_WHEEL"}, {.name = "R_WHEEL"}};
  cnoid::ImuPtr imu_;
  double control_period_;

  double body_pitch_angle_;
  double body_pitch_rate_;

  WheelSet<float> wheel_torque_command_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
  WheelSet<rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr>
      wheel_torque_command_subs_;
  rclcpp::executors::StaticSingleThreadedExecutor::UniquePtr executor_;
};
}  // namespace vnoid_challenge

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(
    vnoid_challenge::InvertedPendulumController)

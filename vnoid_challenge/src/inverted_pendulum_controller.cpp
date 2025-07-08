#include <cnoid/Imu>
#include <cnoid/SimpleController>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>

namespace vnoid_challenge
{
class InvertedPendulumController : public cnoid::SimpleController
{
 public:
  bool configure(cnoid::SimpleControllerConfig* config) override
  {
    node_ = std::make_shared<rclcpp::Node>(config->controllerName());
    wheel_velocity_command_subs_.left()
        = node_->create_subscription<std_msgs::msg::Float32>(
            "wheel_velocity_command/left", 1,
            [this](const std_msgs::msg::Float32::ConstSharedPtr msg)
            { wheel_velocity_command_.left() = msg->data; });
    wheel_velocity_command_subs_.right()
        = node_->create_subscription<std_msgs::msg::Float32>(
            "wheel_velocity_command/right", 1,
            [this](const std_msgs::msg::Float32::ConstSharedPtr msg)
            { wheel_velocity_command_.right() = msg->data; });

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
      wheel.link->setActuationMode(cnoid::Link::JointVelocity);
      io->enableOutput(wheel.link);
    }

    imu_ = io->body()->findDevice<cnoid::Imu>("IMU");
    io->enableInput(imu_);

    return true;
  }

  bool control() override
  {
    executor_->spin_some();

    wheels_.left().link->dq_target() = wheel_velocity_command_.left();
    wheels_.right().link->dq_target() = wheel_velocity_command_.right();

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

  static constexpr double WHEEL_RADIUS = 0.1;
  static constexpr double TREAD = 0.4;

  WheelSet<Link> wheels_{{.name = "L_WHEEL"}, {.name = "R_WHEEL"}};
  cnoid::ImuPtr imu_;

  WheelSet<float> wheel_velocity_command_;

  rclcpp::Node::SharedPtr node_;
  WheelSet<rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr>
      wheel_velocity_command_subs_;
  rclcpp::executors::StaticSingleThreadedExecutor::UniquePtr executor_;
};
}  // namespace vnoid_challenge

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(
    vnoid_challenge::InvertedPendulumController)

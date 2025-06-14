#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cnoid/SimpleController>
#include <mutex>

namespace choreonoid_ros_tutorial
{
class RTTTankController : public cnoid::SimpleController
{
public:
  bool configure(cnoid::SimpleControllerConfig* config) override
  {
    node_ = std::make_shared<rclcpp::Node>(config->controllerName());
    cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 1, std::bind(&RTTTankController::cmd_vel_callback, this, std::placeholders::_1));

    executor_ = std::make_unique<rclcpp::executors::StaticSingleThreadedExecutor>();
    executor_->add_node(node_);

    return true;
  }

  bool initialize(cnoid::SimpleControllerIO* io) override
  {
    for (auto& track : tracks_)
    {
      track.link = io->body()->link(track.name);
      io->enableOutput(track.link, cnoid::SimpleController::StateType::JointVelocity);
    }

    for (auto& [joint, state] : turret_joints_)
    {
      joint.link = io->body()->link(joint.name);
      state.q_ref = state.q_prev = joint.link->q();
      joint.link->setActuationMode(cnoid::SimpleController::StateType::JointTorque);
      io->enableIO(joint.link);
    }

    dt_ = io->timeStep();

    return true;
  }

  bool control() override
  {
    executor_->spin_some();

    // set the velocity of each tracks
    tracks_[LEFT].link->dq_target() = 0.5 * cmd_vel_.linear.x - 0.3 * cmd_vel_.angular.z;
    tracks_[RIGHT].link->dq_target() = 0.5 * cmd_vel_.linear.x + 0.3 * cmd_vel_.angular.z;

    for (auto& [joint, state] : turret_joints_)
    {
      const auto q = joint.link->q();
      const auto dq = (q - state.q_prev) / dt_;
      const auto dq_ref = 0.0;
      joint.link->u() = K_P * (state.q_ref - q) + K_D * (dq_ref - dq);
      state.q_prev = q;
    }

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

  void cmd_vel_callback(const geometry_msgs::msg::Twist::ConstSharedPtr msg)
  {
    cmd_vel_ = *msg;
  }

private:
  enum TrackDirection
  {
    LEFT = 0,
    RIGHT = 1
  };

  struct Link
  {
    cnoid::LinkPtr link;
    std::string name;
  };

  struct JointState
  {
    double q_ref;
    double q_prev;
  };

  static constexpr double K_P = 200.0;
  static constexpr double K_D = 50.0;

  std::array<std::pair<Link, JointState>, 2> turret_joints_{ { { Link{ .name = "TURRET_Y" }, {} },
                                                               { Link{ .name = "TURRET_P" }, {} } } };
  std::array<Link, 2> tracks_{ Link{ .name = "TRACK_L" }, Link{ .name = "TRACK_R" } };
  double dt_;

  geometry_msgs::msg::Twist cmd_vel_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::executors::StaticSingleThreadedExecutor::UniquePtr executor_;
};
}  // namespace choreonoid_ros_tutorial

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(choreonoid_ros_tutorial::RTTTankController)

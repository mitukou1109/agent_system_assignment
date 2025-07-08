#include <cnoid/Imu>
#include <cnoid/Joystick>
#include <cnoid/SimpleController>
#include <iostream>
#include <optional>

class InvertedPendulumController : public cnoid::SimpleController
{
 public:
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
    joystick_.readCurrentState();

    const auto translational_velocity
        = -joystick_.getPosition(cnoid::Joystick::L_STICK_V_AXIS)
          * TRANSLATIONAL_VELOCITY_FACTOR;
    const auto rotational_velocity
        = joystick_.getPosition(cnoid::Joystick::L_STICK_H_AXIS)
          * ROTATIONAL_VELOCITY_FACTOR;

    wheels_[LEFT].link->dq_target()
        = (translational_velocity + TURNING_RADIUS * rotational_velocity)
          / WHEEL_RADIUS;
    wheels_[RIGHT].link->dq_target()
        = (translational_velocity - TURNING_RADIUS * rotational_velocity)
          / WHEEL_RADIUS;

    return true;
  }

 private:
  enum Direction
  {
    LEFT = 0,
    RIGHT = 1
  };

  struct Link
  {
    cnoid::LinkPtr link;
    std::string name;
  };

  static constexpr double TRANSLATIONAL_VELOCITY_FACTOR = 1.0;
  static constexpr double ROTATIONAL_VELOCITY_FACTOR = 1.0;

  static constexpr double WHEEL_RADIUS = 0.1;
  static constexpr double TURNING_RADIUS = 0.2;

  std::array<Link, 2> wheels_{Link{.name = "L_WHEEL"}, Link{.name = "R_WHEEL"}};
  cnoid::Joystick joystick_;

  cnoid::ImuPtr imu_;
};

CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(InvertedPendulumController)

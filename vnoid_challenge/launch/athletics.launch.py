from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    vnoid_challenge_share_dir = FindPackageShare("vnoid_challenge")

    param_file = ParameterFile(
        PathJoinSubstitution([vnoid_challenge_share_dir, "config", "athletics.yaml"]),
        allow_substs=True,
    )

    project_file = PathJoinSubstitution(
        [vnoid_challenge_share_dir, "project", "athletics.cnoid"]
    )

    set_choreonoid_env_var = SetEnvironmentVariable("CNOID_USE_GLSL", "0")

    choreonoid_node = Node(
        package="choreonoid_ros",
        executable="choreonoid",
        name="choreonoid",
        output="screen",
        parameters=[param_file],
        arguments=[project_file],
    )

    balance_controller_node = Node(
        package="vnoid_challenge",
        executable="balance_controller_node",
        name="balance_controller",
        output="both",
        parameters=[param_file],
    )

    return LaunchDescription(
        [
            set_choreonoid_env_var,
            choreonoid_node,
            balance_controller_node,
        ]
    )

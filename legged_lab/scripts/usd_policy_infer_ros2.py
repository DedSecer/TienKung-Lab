# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# ROS2 configuration constants
ROS2_DOMAIN_ID = 0

# Camera configuration constants
CAMERA_CONFIG = {
    "rgb_topic": "/camera/color/image_rect_color",
    "depth_topic": "/depth",
    "camera_info_topic": "/camera_info",
    "frame_id": "robot_camera",
    "width": 640,
    "height": 480,
}

# PhysX LiDAR configuration constants
LIDAR_CONFIG = {
    "topic": "/point_cloud",
    "frame_id": "lidar_frame",
    "fov": (360.0, 30.0),
    "resolution": (0.4, 4.0),
    "rotation_rate": 20.0,
    "valid_range": (0.4, 100.0),
    "high_lod": True,
}

# ROS2 cmd_vel subscriber configuration constants
CMD_VEL_CONFIG = {
    "topic": "/cmd_vel",
    "max_lin_vel_x": 1.0,
    "max_lin_vel_y": 0.5,
    "max_ang_vel_z": 1.57,
    "lin_vel_gain": 1.0,
    "ang_vel_gain": 1.0,
}

# High-frequency IMU publisher configuration constants
IMU_CONFIG = {
    "topic": "/imu/data",
    "frame_id": "imu_link",
    "publish_rate": 60.0,
}

# Odom TF publisher configuration constants
ODOM_TF_CONFIG = {
    "topic": "/tf",
    "odom_frame_id": "odom",
    "base_frame_id": "base_link",
    "publish_rate": 60.0,
}

# Clock publisher configuration constants
CLOCK_CONFIG = {
    "topic": "/clock",
    "publish_rate": 100.0,
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Policy inference for TienKung robot in a USD environment with ROS2 camera publishing.")
parser.add_argument("--task", type=str, default="walk", help="Name of the task.")
parser.add_argument("--policy_path", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--usd_path", type=str, default="../sense/museum/museum.usd", help="Path to custom USD environment file (default: ../sense/museum/museum.usd).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Enable the ROS2 bridge extension before launching the app
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import torch

import omni.usd
import omni.timeline
import rclpy
from pxr import UsdGeom

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.infer.common import load_policy, configure_env_for_usd, prepare_usd_stage
from legged_lab.utils.infer.sensor_setup import create_camera_on_robot, create_physx_lidar_on_robot
from legged_lab.utils.infer.omnigraph_setup import (
    enable_required_extensions,
    setup_ros2_camera_graph,
    setup_ros2_physx_lidar_graph,
)
from legged_lab.utils.infer.ros2_publishers import (
    HighFreqImuPublisher,
    HighFreqLidarPublisher,
    ClockPublisher,
    OdomTFPublisher,
)
from legged_lab.utils.infer.ros2_subscribers import CmdVelSubscriber

# Enable required extensions for ROS2 camera publishing
extensions_enabled = enable_required_extensions()
simulation_app.update()  # Update to ensure extensions are loaded


def main():
    """Main function."""
    # Track resources for cleanup
    temp_usd_path = None
    cmd_vel_subscriber = None
    imu_publisher = None
    odom_tf_publisher = None
    clock_publisher = None
    env = None

    # load the trained jit policy
    policy, device = load_policy(args_cli.policy_path)

    # get environment configuration
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # prepare cleaned USD stage
    usd_path = os.path.abspath(args_cli.usd_path)
    temp_usd_path = prepare_usd_stage(usd_path)

    # configure environment for USD inference
    configure_env_for_usd(env_cfg, temp_usd_path, num_envs=1,
                            seed=args_cli.seed, device=device)

    # create environment
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    print(f"[INFO] Created environment: {env_class_name}")

    # Get the current stage after environment creation
    current_stage = omni.usd.get_context().get_stage()

    # Find the robot prim path - typically under /World/envs/env_0/Robot
    robot_prim_path = "/World/envs/env_0/Robot"
    robot_prim = current_stage.GetPrimAtPath(robot_prim_path)

    # Create camera on the robot
    camera_path = create_camera_on_robot(
        stage=current_stage,
        robot_prim_path=robot_prim_path,
        camera_name="head_camera",
        local_position=(0.3, 0.0, 0.65),
        local_rotation=(90.0, -90.0, 0.0),
        width=CAMERA_CONFIG["width"],
        height=CAMERA_CONFIG["height"]
    )

    # Setup ROS2 camera publishing graph
    ros2_graph = setup_ros2_camera_graph(
        camera_prim_path=camera_path,
        rgb_topic=CAMERA_CONFIG["rgb_topic"],
        depth_topic=CAMERA_CONFIG["depth_topic"],
        camera_info_topic=CAMERA_CONFIG["camera_info_topic"],
        frame_id=CAMERA_CONFIG["frame_id"],
        width=CAMERA_CONFIG["width"],
        height=CAMERA_CONFIG["height"],
        domain_id=ROS2_DOMAIN_ID
    )
    print("[INFO] ROS2 camera publishing enabled successfully!")
    print(f"[INFO] Topics: {CAMERA_CONFIG['rgb_topic']}, {CAMERA_CONFIG['depth_topic']}, {CAMERA_CONFIG['camera_info_topic']}")
    print(f"[INFO] To view topics, run: ros2 topic list")
    print(f"[INFO] To view RGB image: ros2 run rqt_image_view rqt_image_view {CAMERA_CONFIG['rgb_topic']}")

    # Create PhysX LiDAR on the robot
    lidar_path = create_physx_lidar_on_robot(
        stage=current_stage,
        robot_prim_path=robot_prim_path,
        lidar_name="mid360_lidar",
        local_position=(0.0, 0.0, 1.0),
        local_rotation=(0.0, 0.0, 0.0),
        fov=LIDAR_CONFIG["fov"],
        resolution=LIDAR_CONFIG["resolution"],
        rotation_rate=LIDAR_CONFIG["rotation_rate"],
        valid_range=LIDAR_CONFIG["valid_range"],
        high_lod=LIDAR_CONFIG["high_lod"],
    )

    # Setup ROS2 LiDAR publishing graph
    ros2_lidar_graph = setup_ros2_physx_lidar_graph(
        lidar_prim_path=lidar_path,
        point_cloud_topic=LIDAR_CONFIG["topic"],
        frame_id=LIDAR_CONFIG["frame_id"],
        domain_id=ROS2_DOMAIN_ID
    )
    print("[INFO] ROS2 PhysX LiDAR publishing enabled successfully!")
    print(f"[INFO] LiDAR Point Cloud topic: {LIDAR_CONFIG['topic']}")
    print(f"[INFO] To view point cloud: ros2 topic echo {LIDAR_CONFIG['topic']}")
    print(f"[INFO] To visualize in RViz2: Add PointCloud2 display with topic {LIDAR_CONFIG['topic']}")

    # Setup ROS2 cmd_vel subscriber
    cmd_vel_subscriber = CmdVelSubscriber(
        topic_name=CMD_VEL_CONFIG["topic"],
        max_lin_vel_x=CMD_VEL_CONFIG["max_lin_vel_x"],
        max_lin_vel_y=CMD_VEL_CONFIG["max_lin_vel_y"],
        max_ang_vel_z=CMD_VEL_CONFIG["max_ang_vel_z"],
        domain_id=ROS2_DOMAIN_ID
    )
    print("[INFO] ROS2 cmd_vel subscriber enabled successfully!")
    print(f"[INFO] Subscribing to topic: {CMD_VEL_CONFIG['topic']}")
    print(f"[INFO] To send velocity commands: ros2 topic pub {CMD_VEL_CONFIG['topic']} geometry_msgs/msg/Twist '{{linear: {{x: 0.5, y: 0.0, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.2}}}}'")
    print(f"[INFO] Or use teleop_twist_keyboard: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:={CMD_VEL_CONFIG['topic']}")

    # Setup high-frequency IMU publisher
    imu_publisher = HighFreqImuPublisher(
        topic_name=IMU_CONFIG["topic"],
        frame_id=IMU_CONFIG["frame_id"],
        publish_rate=IMU_CONFIG["publish_rate"],
        domain_id=ROS2_DOMAIN_ID
    )
    print("[INFO] High-frequency IMU publisher enabled successfully!")
    print(f"[INFO] Publishing to topic: {IMU_CONFIG['topic']} at {IMU_CONFIG['publish_rate']} Hz")
    print(f"[INFO] To check IMU frequency: ros2 topic hz {IMU_CONFIG['topic']}")

    # Setup odom TF publisher
    odom_tf_publisher = OdomTFPublisher(
        topic_name=ODOM_TF_CONFIG["topic"],
        odom_frame_id=ODOM_TF_CONFIG["odom_frame_id"],
        base_frame_id=ODOM_TF_CONFIG["base_frame_id"],
        publish_rate=ODOM_TF_CONFIG["publish_rate"],
        domain_id=ROS2_DOMAIN_ID
    )
    print("[INFO] Odom TF publisher enabled successfully!")
    print(f"[INFO] Publishing TF {ODOM_TF_CONFIG['odom_frame_id']} -> {ODOM_TF_CONFIG['base_frame_id']} at {ODOM_TF_CONFIG['publish_rate']} Hz")
    print(f"[INFO] To view TF tree: ros2 run tf2_tools view_frames")

    # Setup clock publisher
    clock_publisher = ClockPublisher(
        topic_name=CLOCK_CONFIG["topic"],
        publish_rate=CLOCK_CONFIG["publish_rate"],
        domain_id=ROS2_DOMAIN_ID
    )
    print("[INFO] Clock publisher enabled successfully!")
    print(f"[INFO] Publishing simulation time to topic: {CLOCK_CONFIG['topic']} at {CLOCK_CONFIG['publish_rate']} Hz")
    print("[INFO] ROS2 nodes should use 'use_sim_time:=true' to synchronize with simulation")

    # setup keyboard control if not headless
    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841
        print("[INFO] Keyboard control enabled. Use arrow keys to control the robot.")

    # run inference with the policy
    obs, _ = env.get_observations()
    print("[INFO] Starting policy inference...")
    print("[INFO] Press Ctrl+C to stop the simulation.")

    with torch.inference_mode():
        while simulation_app.is_running():
            # Update velocity commands from cmd_vel subscriber
            if cmd_vel_subscriber is not None:
                lin_vel_x, lin_vel_y, ang_vel_z = cmd_vel_subscriber.get_velocity_command()

                # Apply gains to improve responsiveness for Nav2
                # Nav2 often outputs small velocities that RL policies might ignore
                lin_vel_x *= CMD_VEL_CONFIG["lin_vel_gain"]
                lin_vel_y *= CMD_VEL_CONFIG["lin_vel_gain"]
                ang_vel_z *= CMD_VEL_CONFIG["ang_vel_gain"]

                # Simple deadzone to avoid drift
                if abs(lin_vel_x) < 0.01: lin_vel_x = 0.0
                if abs(lin_vel_y) < 0.01: lin_vel_y = 0.0
                if abs(ang_vel_z) < 0.01: ang_vel_z = 0.0

                # Trick: Some policies struggle to turn in place without forward motion.
                # If we have rotation but no linear velocity, inject a tiny forward surge
                # to "wake up" the stepping controller.
                if abs(ang_vel_z) > 0.05 and abs(lin_vel_x) < 0.01 and abs(lin_vel_y) < 0.01:
                    lin_vel_x = 0.001

                # Update the command generator's command tensor
                # command tensor shape: (num_envs, 3) where [lin_vel_x, lin_vel_y, ang_vel_z]
                env.command_generator.command[:, 0] = lin_vel_x
                env.command_generator.command[:, 1] = lin_vel_y
                env.command_generator.command[:, 2] = ang_vel_z

            action = policy(obs)
            obs, _, _, _ = env.step(action)

            # Update high-frequency IMU publisher with current robot state
            if imu_publisher is not None:
                # Get robot state data for IMU
                robot = env.robot
                # Angular velocity in body frame
                ang_vel = robot.data.root_ang_vel_b[0]  # Shape: (3,) for first env
                # Linear velocity in body frame
                lin_vel = robot.data.root_lin_vel_b[0]  # Shape: (3,) for first env
                # Orientation quaternion (w, x, y, z) - Isaac Sim convention
                orientation = robot.data.root_quat_w[0]  # Shape: (4,) for first env
                # Projected gravity in body frame (for acceleration compensation)
                gravity = robot.data.projected_gravity_b[0]  # Shape: (3,) for first env

                # Get simulation time from Isaac Sim timeline (same time source as LiDAR)
                sim_time = omni.timeline.get_timeline_interface().get_current_time()

                imu_publisher.update_imu_data(
                    ang_vel=ang_vel,
                    lin_vel=lin_vel,
                    orientation=orientation,
                    gravity=gravity * 9.81,  # Scale to m/s^2 (projected_gravity_b is normalized)
                    sim_time=sim_time  # Pass simulation time for LiDAR synchronization
                )

            # Update odom TF publisher with current robot pose
            if odom_tf_publisher is not None:
                # Get robot state data for odom TF
                robot = env.robot
                # Position in world frame
                position = robot.data.root_pos_w[0]  # Shape: (3,) for first env
                # Orientation quaternion (w, x, y, z) - Isaac Sim convention
                orientation = robot.data.root_quat_w[0]  # Shape: (4,) for first env

                # Get simulation time
                sim_time = omni.timeline.get_timeline_interface().get_current_time()

                odom_tf_publisher.update_robot_pose(
                    position=position,
                    orientation=orientation,
                    sim_time=sim_time
                )

            # Update clock publisher with current simulation time
            if clock_publisher is not None:
                sim_time = omni.timeline.get_timeline_interface().get_current_time()
                clock_publisher.update_sim_time(sim_time)


if __name__ == "__main__":
    main()
    simulation_app.close()

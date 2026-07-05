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

# add argparse arguments
parser = argparse.ArgumentParser(description="Policy inference for TienKung robot in a USD environment with ROS2 camera publishing.")
parser.add_argument("--task", type=str, default="walk", help="Name of the task.")
parser.add_argument("--policy_path", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--usd_path", type=str, default="../sense/museum/museum.usd", help="Path to custom USD environment file (default: ../sense/museum/museum.usd).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# ROS2 camera configuration
parser.add_argument("--rgb_topic", type=str, default="/camera/color/image_rect_color", help="ROS2 topic name for RGB image.")
parser.add_argument("--depth_topic", type=str, default="/depth", help="ROS2 topic name for depth image.")
parser.add_argument("--camera_info_topic", type=str, default="/camera_info", help="ROS2 topic name for camera info.")
parser.add_argument("--camera_frame_id", type=str, default="robot_camera", help="Frame ID for camera messages.")
parser.add_argument("--camera_width", type=int, default=640, help="Camera image width.")
parser.add_argument("--camera_height", type=int, default=480, help="Camera image height.")
parser.add_argument("--ros2_domain_id", type=int, default=0, help="ROS2 domain ID.")
# PhysX LiDAR configuration
parser.add_argument("--enable_lidar", action=argparse.BooleanOptionalAction, default=True, help="Enable PhysX LiDAR sensor (enabled by default, use --no-enable_lidar to disable).")
parser.add_argument("--lidar_topic", type=str, default="/point_cloud", help="ROS2 topic name for LiDAR point cloud.")
parser.add_argument("--lidar_frame_id", type=str, default="lidar_frame", help="Frame ID for LiDAR messages.")
parser.add_argument("--lidar_fov", type=float, nargs=2, default=[360.0, 30.0], help="PhysX LiDAR field of view [horizontal, vertical] in degrees (default: 360.0 30.0).")
parser.add_argument("--lidar_resolution", type=float, nargs=2, default=[0.4, 4.0], help="PhysX LiDAR resolution [horizontal, vertical] in degrees (default: 0.4 4.0).")
parser.add_argument("--lidar_rotation_rate", type=float, default=20.0, help="PhysX LiDAR rotation rate in Hz (default: 20.0). Set to 0 for static scan.")
parser.add_argument("--lidar_valid_range", type=float, nargs=2, default=[0.4, 100.0], help="PhysX LiDAR valid range [min, max] in meters (default: 0.4 100.0).")
parser.add_argument("--lidar_high_lod", action="store_true", default=True, help="Enable high LOD for 3D point cloud output (default: True).")
# ROS2 cmd_vel subscriber configuration
parser.add_argument("--enable_cmd_vel", action="store_true", help="Enable ROS2 cmd_vel subscriber for velocity control.")
parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS2 topic name for velocity commands (geometry_msgs/Twist).")
parser.add_argument("--max_lin_vel_x", type=float, default=1.0, help="Maximum linear velocity in x direction (m/s).")
parser.add_argument("--max_lin_vel_y", type=float, default=0.5, help="Maximum linear velocity in y direction (m/s).")
parser.add_argument("--max_ang_vel_z", type=float, default=1.57, help="Maximum angular velocity around z axis (rad/s).")
parser.add_argument("--lin_vel_gain", type=float, default=1.0, help="Gain multiplier for linear velocity commands (default: 1.0). Increase for more responsive movement.")
parser.add_argument("--ang_vel_gain", type=float, default=1.0, help="Gain multiplier for angular velocity commands (default: 1.0). Increase for fuller turns.")
# High-frequency IMU publisher configuration
parser.add_argument("--enable_high_freq_imu", action=argparse.BooleanOptionalAction, default=True, help="Enable high-frequency IMU publisher (enabled by default, use --no-enable_high_freq_imu to disable).")
parser.add_argument("--imu_topic", type=str, default="/imu/data", help="ROS2 topic name for high-frequency IMU data.")
parser.add_argument("--imu_frame_id", type=str, default="imu_link", help="Frame ID for IMU messages.")
parser.add_argument("--imu_publish_rate", type=float, default=60.0, help="IMU publish rate in Hz (default: 60Hz).")
# Odom TF publisher configuration
parser.add_argument("--enable_odom_tf", action=argparse.BooleanOptionalAction, default=True, help="Enable odom->base_link TF publisher (enabled by default, use --no-enable_odom_tf to disable).")
parser.add_argument("--odom_tf_topic", type=str, default="/tf", help="ROS2 topic name for odom TF (geometry_msgs/TransformStamped).")
parser.add_argument("--odom_frame_id", type=str, default="odom", help="Frame ID for odom frame.")
parser.add_argument("--base_frame_id", type=str, default="base_link", help="Frame ID for robot base frame.")
parser.add_argument("--odom_tf_publish_rate", type=float, default=60.0, help="Odom TF publish rate in Hz (default: 60Hz).")
# Clock publisher configuration
parser.add_argument("--enable_clock", action=argparse.BooleanOptionalAction, default=True, help="Enable /clock topic publisher (enabled by default, use --no-enable_clock to disable).")
parser.add_argument("--clock_topic", type=str, default="/clock", help="ROS2 topic name for simulation clock.")
parser.add_argument("--clock_publish_rate", type=float, default=100.0, help="Clock publish rate in Hz (default: 100Hz).")

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

    try:
        # load the trained jit policy
        policy, device = load_policy(args_cli.policy_path)

        # get environment configuration
        env_class_name = args_cli.task
        env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

        # prepare cleaned USD stage
        usd_path = os.path.abspath(args_cli.usd_path)
        temp_usd_path = prepare_usd_stage(usd_path)

        # configure environment for USD inference
        configure_env_for_usd(env_cfg, temp_usd_path, num_envs=args_cli.num_envs,
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

        if not robot_prim.IsValid():
            print(f"[WARN] Robot not found at {robot_prim_path}, searching for alternative paths...")
            # Search for robot in stage
            for prim in current_stage.Traverse():
                if "Robot" in prim.GetPath().pathString and prim.IsA(UsdGeom.Xform):
                    robot_prim_path = prim.GetPath().pathString
                    print(f"[INFO] Found robot at: {robot_prim_path}")
                    break

        # Create camera on the robot
        camera_path = create_camera_on_robot(
            stage=current_stage,
            robot_prim_path=robot_prim_path,
            camera_name="head_camera",
            local_position=(0.3, 0.0, 0.65),  # 前方 0.3m, 抬高 0.4m
            local_rotation=(90.0, -90.0, 0.0),  # 绕 Y 轴旋转 90度，相机朝前
            width=args_cli.camera_width,
            height=args_cli.camera_height
        )

        # Update simulation to initialize the camera
        simulation_app.update()

        if camera_path:
            # Setup ROS2 camera publishing graph
            try:
                ros2_graph = setup_ros2_camera_graph(
                    camera_prim_path=camera_path,
                    rgb_topic=args_cli.rgb_topic,
                    depth_topic=args_cli.depth_topic,
                    camera_info_topic=args_cli.camera_info_topic,
                    frame_id=args_cli.camera_frame_id,
                    width=args_cli.camera_width,
                    height=args_cli.camera_height,
                    domain_id=args_cli.ros2_domain_id
                )
                print("[INFO] ROS2 camera publishing enabled successfully!")
                print(f"[INFO] Topics: {args_cli.rgb_topic}, {args_cli.depth_topic}, {args_cli.camera_info_topic}")
                print(f"[INFO] To view topics, run: ros2 topic list")
                print(f"[INFO] To view RGB image: ros2 run rqt_image_view rqt_image_view {args_cli.rgb_topic}")
            except Exception as e:
                print(f"[ERROR] Failed to setup ROS2 camera graph: {e}")
                print("[WARN] Continuing without ROS2 camera publishing...")
        else:
            print("[WARN] Camera creation failed, skipping ROS2 camera publishing setup")

        # Create PhysX LiDAR on the robot if enabled
        if args_cli.enable_lidar:
            lidar_path = create_physx_lidar_on_robot(
                stage=current_stage,
                robot_prim_path=robot_prim_path,
                lidar_name="mid360_lidar",
                local_position=(0.0, 0.0, 1.0),  # 机器人 pelvis 上方 1.0m
                local_rotation=(0.0, 0.0, 0.0),
                fov=tuple(args_cli.lidar_fov),
                resolution=tuple(args_cli.lidar_resolution),
                rotation_rate=args_cli.lidar_rotation_rate,
                valid_range=tuple(args_cli.lidar_valid_range),
                high_lod=args_cli.lidar_high_lod,
            )

            # Update simulation to initialize the lidar
            simulation_app.update()

            if lidar_path:
                # Setup ROS2 LiDAR publishing graph
                try:
                    ros2_lidar_graph = setup_ros2_physx_lidar_graph(
                        lidar_prim_path=lidar_path,
                        point_cloud_topic=args_cli.lidar_topic,
                        frame_id=args_cli.lidar_frame_id,
                        domain_id=args_cli.ros2_domain_id
                    )
                    print("[INFO] ROS2 PhysX LiDAR publishing enabled successfully!")
                    print(f"[INFO] LiDAR Point Cloud topic: {args_cli.lidar_topic}")
                    print(f"[INFO] To view point cloud: ros2 topic echo {args_cli.lidar_topic}")
                    print(f"[INFO] To visualize in RViz2: Add PointCloud2 display with topic {args_cli.lidar_topic}")
                except Exception as e:
                    print(f"[ERROR] Failed to setup ROS2 PhysX LiDAR graph: {e}")
                    print("[WARN] Continuing without ROS2 LiDAR publishing...")
            else:
                print("[WARN] PhysX LiDAR creation failed, skipping ROS2 LiDAR publishing setup")
        else:
            print("[INFO] LiDAR disabled. Use --enable_lidar to enable PhysX LiDAR sensor.")

        # Setup ROS2 cmd_vel subscriber if enabled
        if args_cli.enable_cmd_vel:
            try:
                cmd_vel_subscriber = CmdVelSubscriber(
                    topic_name=args_cli.cmd_vel_topic,
                    max_lin_vel_x=args_cli.max_lin_vel_x,
                    max_lin_vel_y=args_cli.max_lin_vel_y,
                    max_ang_vel_z=args_cli.max_ang_vel_z,
                    domain_id=args_cli.ros2_domain_id
                )
                print("[INFO] ROS2 cmd_vel subscriber enabled successfully!")
                print(f"[INFO] Subscribing to topic: {args_cli.cmd_vel_topic}")
                print(f"[INFO] To send velocity commands: ros2 topic pub {args_cli.cmd_vel_topic} geometry_msgs/msg/Twist '{{linear: {{x: 0.5, y: 0.0, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.2}}}}'")
                print(f"[INFO] Or use teleop_twist_keyboard: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:={args_cli.cmd_vel_topic}")
            except Exception as e:
                print(f"[ERROR] Failed to setup cmd_vel subscriber: {e}")
                print("[WARN] Continuing without cmd_vel control...")
        else:
            print("[INFO] cmd_vel subscriber disabled. Use --enable_cmd_vel to enable velocity control via ROS2.")

        # Setup high-frequency IMU publisher if enabled
        if args_cli.enable_high_freq_imu:
            try:
                imu_publisher = HighFreqImuPublisher(
                    topic_name=args_cli.imu_topic,
                    frame_id=args_cli.imu_frame_id,
                    publish_rate=args_cli.imu_publish_rate,
                    domain_id=args_cli.ros2_domain_id
                )
                print("[INFO] High-frequency IMU publisher enabled successfully!")
                print(f"[INFO] Publishing to topic: {args_cli.imu_topic} at {args_cli.imu_publish_rate} Hz")
                print(f"[INFO] To check IMU frequency: ros2 topic hz {args_cli.imu_topic}")
            except Exception as e:
                print(f"[ERROR] Failed to setup high-frequency IMU publisher: {e}")
                print("[WARN] Continuing without high-frequency IMU publishing...")
        else:
            print("[INFO] High-frequency IMU publisher disabled. Use --enable_high_freq_imu to enable.")

        # Setup odom TF publisher if enabled
        if args_cli.enable_odom_tf:
            try:
                odom_tf_publisher = OdomTFPublisher(
                    topic_name=args_cli.odom_tf_topic,
                    odom_frame_id=args_cli.odom_frame_id,
                    base_frame_id=args_cli.base_frame_id,
                    publish_rate=args_cli.odom_tf_publish_rate,
                    domain_id=args_cli.ros2_domain_id
                )
                print("[INFO] Odom TF publisher enabled successfully!")
                print(f"[INFO] Publishing TF {args_cli.odom_frame_id} -> {args_cli.base_frame_id} at {args_cli.odom_tf_publish_rate} Hz")
                print(f"[INFO] To view TF tree: ros2 run tf2_tools view_frames")
            except Exception as e:
                print(f"[ERROR] Failed to setup odom TF publisher: {e}")
                print("[WARN] Continuing without odom TF publishing...")
        else:
            print("[INFO] Odom TF publisher disabled. Use --enable_odom_tf to enable.")

        # Setup clock publisher if enabled
        if args_cli.enable_clock:
            try:
                clock_publisher = ClockPublisher(
                    topic_name=args_cli.clock_topic,
                    publish_rate=args_cli.clock_publish_rate,
                    domain_id=args_cli.ros2_domain_id
                )
                print("[INFO] Clock publisher enabled successfully!")
                print(f"[INFO] Publishing simulation time to topic: {args_cli.clock_topic} at {args_cli.clock_publish_rate} Hz")
                print("[INFO] ROS2 nodes should use 'use_sim_time:=true' to synchronize with simulation")
            except Exception as e:
                print(f"[ERROR] Failed to setup clock publisher: {e}")
                print("[WARN] Continuing without clock publishing...")
        else:
            print("[INFO] Clock publisher disabled. Use --enable_clock to enable.")

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
                    lin_vel_x *= args_cli.lin_vel_gain
                    lin_vel_y *= args_cli.lin_vel_gain
                    ang_vel_z *= args_cli.ang_vel_gain

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

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up resources...")

        # Cleanup ROS2 publishers/subscribers
        if cmd_vel_subscriber is not None:
            try:
                cmd_vel_subscriber.shutdown()
            except Exception as e:
                print(f"[WARN] Error shutting down cmd_vel subscriber: {e}")

        if imu_publisher is not None:
            try:
                imu_publisher.shutdown()
            except Exception as e:
                print(f"[WARN] Error shutting down IMU publisher: {e}")

        if odom_tf_publisher is not None:
            try:
                odom_tf_publisher.shutdown()
            except Exception as e:
                print(f"[WARN] Error shutting down odom TF publisher: {e}")

        if clock_publisher is not None:
            try:
                clock_publisher.shutdown()
            except Exception as e:
                print(f"[WARN] Error shutting down clock publisher: {e}")

        # Shutdown rclpy if it was initialized
        try:
            if rclpy.ok():
                rclpy.shutdown()
                print("[INFO] rclpy shutdown complete")
        except Exception as e:
            print(f"[WARN] Error shutting down rclpy: {e}")

        # Clean up temporary USD file
        if temp_usd_path is not None:
            try:
                if os.path.exists(temp_usd_path):
                    os.remove(temp_usd_path)
                    print(f"[INFO] Cleaned up temporary USD file: {temp_usd_path}")
            except Exception as e:
                print(f"[WARN] Failed to delete temporary USD file {temp_usd_path}: {e}")

        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
    simulation_app.close()

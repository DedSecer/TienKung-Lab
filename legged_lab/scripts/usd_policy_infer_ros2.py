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

"""
This script demonstrates policy inference in a prebuilt USD environment for TienKung robot
with camera sensors and ROS2 topic publishing.

In this example, we use a locomotion policy to control the TienKung robot. The robot was trained
using the walk task. The robot is commanded to move forward at a constant velocity.
Additionally, camera sensors are added to the robot and their data is published to ROS2 topics.

Prerequisites:
    - ROS 2 must be installed and sourced before launching Isaac Sim
    - The isaacsim.ros2.bridge extension must be enabled

Usage:
    # Run with default museum USD environment (../sense/museum/museum.usd)
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/exported/policy.pt

    # Run with custom USD environment
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/exported/policy.pt --usd_path /path/to/custom.usd

    # Run with custom camera topic names
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/policy.pt --rgb_topic /camera/rgb --depth_topic /camera/depth

    # Run with RTX LiDAR disabled (LiDAR is enabled by default)
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/policy.pt --no-enable_lidar

    # Run with cmd_vel subscriber enabled for velocity control
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/policy.pt --enable_cmd_vel --cmd_vel_topic /cmd_vel

    # Run with minimal ROS2 features (disable IMU, clock, odom TF, LiDAR)
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/policy.pt --no-enable_lidar --no-enable_high_freq_imu --no-enable_clock --no-enable_odom_tf

ROS2 Topics Published (by default):
    - /rgb (sensor_msgs/Image): RGB camera image
    - /depth (sensor_msgs/Image): Depth camera image
    - /camera_info (sensor_msgs/CameraInfo): Camera intrinsic parameters
    - /point_cloud (sensor_msgs/PointCloud2): RTX LiDAR point cloud data (enabled by default, use --no-enable_lidar to disable)
    - /imu/data (sensor_msgs/Imu): High-frequency IMU data (enabled by default, use --no-enable_high_freq_imu to disable)
    - /clock (rosgraph_msgs/Clock): Simulation clock (enabled by default, use --no-enable_clock to disable)
    - /tf (tf2_msgs/TFMessage): odom->base_link transform (enabled by default, use --no-enable_odom_tf to disable)

ROS2 Topics Subscribed:
    - /cmd_vel (geometry_msgs/Twist): Velocity commands for robot control (disabled by default, use --enable_cmd_vel to enable)
        - linear.x: Forward/backward velocity (m/s)
        - linear.y: Left/right velocity (m/s)
        - angular.z: Rotation velocity (rad/s)

"""

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
# RTX LiDAR configuration
parser.add_argument("--enable_lidar", action=argparse.BooleanOptionalAction, default=True, help="Enable RTX LiDAR sensor (enabled by default, use --no-enable_lidar to disable).")
parser.add_argument("--lidar_topic", type=str, default="/point_cloud", help="ROS2 topic name for LiDAR point cloud.")
parser.add_argument("--lidar_frame_id", type=str, default="lidar_frame", help="Frame ID for LiDAR messages.")
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
import io
import os
import torch
import numpy as np

import omni
import omni.graph.core as og
import omni.usd
import omni.kit.app
import omni.timeline
from pxr import Usd, UsdGeom, Gf

# ROS2 imports for cmd_vel subscriber, IMU publisher, and LiDAR publisher
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Imu as ImuMsg
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from rosgraph_msgs.msg import Clock
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    import threading
    import time as time_module
    import struct
    ROS2_AVAILABLE = True
except ImportError:
    print("[WARN] rclpy not available. cmd_vel subscriber, IMU publisher and LiDAR publisher will be disabled.")
    ROS2_AVAILABLE = False

from legged_lab.envs import *  # noqa:F401, F403


class CmdVelSubscriber:
    """
    ROS2 subscriber for cmd_vel topic (geometry_msgs/Twist).
    
    This class subscribes to velocity commands from ROS2 and stores them
    for use in controlling the robot's movement.
    
    The subscriber runs in a separate thread to avoid blocking the simulation.
    """
    
    def __init__(self, topic_name: str = "/cmd_vel", 
                 max_lin_vel_x: float = 1.0,
                 max_lin_vel_y: float = 0.5,
                 max_ang_vel_z: float = 1.0,
                 domain_id: int = 0):
        """
        Initialize the cmd_vel subscriber.
        
        Args:
            topic_name: ROS2 topic name for velocity commands
            max_lin_vel_x: Maximum linear velocity in x direction (m/s)
            max_lin_vel_y: Maximum linear velocity in y direction (m/s)
            max_ang_vel_z: Maximum angular velocity around z axis (rad/s)
            domain_id: ROS2 domain ID
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not available. Cannot create CmdVelSubscriber.")
        
        self.topic_name = topic_name
        self.max_lin_vel_x = max_lin_vel_x
        self.max_lin_vel_y = max_lin_vel_y
        self.max_ang_vel_z = max_ang_vel_z
        
        # Initialize velocity commands to zero
        self._lin_vel_x = 0.0
        self._lin_vel_y = 0.0
        self._ang_vel_z = 0.0
        self._lock = threading.Lock()
        
        # Set ROS_DOMAIN_ID if not already set
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain_id))
        
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS2 node and subscriber
        self._node = rclpy.create_node('isaacsim_cmd_vel_subscriber')
        self._subscription = self._node.create_subscription(
            Twist,
            topic_name,
            self._cmd_vel_callback,
            10  # QoS profile depth
        )
        
        # Create executor and run in separate thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(target=self._spin_thread, daemon=True)
        self._thread.start()
        
        print(f"[INFO] CmdVelSubscriber initialized on topic: {topic_name}")
        print(f"[INFO] Velocity limits: lin_vel_x={max_lin_vel_x}, lin_vel_y={max_lin_vel_y}, ang_vel_z={max_ang_vel_z}")
    
    def _cmd_vel_callback(self, msg: Twist):
        """Callback function for cmd_vel messages."""
        with self._lock:
            # Clamp velocities to maximum values
            self._lin_vel_x = max(-self.max_lin_vel_x, min(self.max_lin_vel_x, msg.linear.x))
            self._lin_vel_y = max(-self.max_lin_vel_y, min(self.max_lin_vel_y, msg.linear.y))
            self._ang_vel_z = max(-self.max_ang_vel_z, min(self.max_ang_vel_z, msg.angular.z))
    
    def _spin_thread(self):
        """Thread function to spin the ROS2 node."""
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)
    
    def get_velocity_command(self) -> tuple:
        """
        Get the current velocity command.
        
        Returns:
            Tuple of (lin_vel_x, lin_vel_y, ang_vel_z)
        """
        with self._lock:
            return (self._lin_vel_x, self._lin_vel_y, self._ang_vel_z)
    
    def shutdown(self):
        """Shutdown the subscriber and cleanup resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self._node:
            self._node.destroy_node()
        
        print("[INFO] CmdVelSubscriber shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass


class HighFreqImuPublisher:
    """
    High-frequency IMU publisher for ROS2.
    
    This class publishes IMU data (angular velocity, linear acceleration, orientation)
    at a configurable high frequency, directly from Isaac Sim robot state data.
    This is needed for SLAM algorithms like FAST-LIO and Point-LIO which require
    high-frequency IMU data (typically 100-400 Hz).
    
    The publisher runs in a separate thread and interpolates data between
    simulation steps to achieve the target publish rate.
    
    IMPORTANT: Uses simulation time instead of wall clock time to ensure
    synchronization with LiDAR timestamps from Isaac Sim ROS2 bridge.
    """
    
    def __init__(self, topic_name: str = "/imu/data", 
                 frame_id: str = "imu_link",
                 publish_rate: float = 200.0,
                 domain_id: int = 0):
        """
        Initialize the high-frequency IMU publisher.
        
        Args:
            topic_name: ROS2 topic name for IMU data
            frame_id: Frame ID for IMU messages
            publish_rate: Target publish rate in Hz
            domain_id: ROS2 domain ID
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not available. Cannot create HighFreqImuPublisher.")
        
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.publish_rate = publish_rate
        self.publish_period = 1.0 / publish_rate
        
        # IMU data storage (thread-safe)
        self._lock = threading.Lock()
        self._ang_vel = np.zeros(3)  # Angular velocity (rad/s) in body frame
        self._lin_acc = np.zeros(3)  # Linear acceleration (m/s^2) in body frame
        self._orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)
        
        # Simulation time tracking (for synchronization with LiDAR)
        self._sim_time = 0.0  # Current simulation time in seconds
        self._sim_time_updated = False  # Flag to indicate new data is available
        self._last_published_sim_time = -1.0  # Last published simulation time
        
        # For numerical differentiation of linear velocity to get acceleration
        self._prev_lin_vel = np.zeros(3)
        self._prev_sim_time = 0.0  # Use simulation time for differentiation
        
        # Set ROS_DOMAIN_ID if not already set
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain_id))
        
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS2 node and publisher with best-effort QoS for high frequency
        self._node = rclpy.create_node('isaacsim_imu_publisher')
        
        # Use reliable QoS for compatibility with SLAM algorithms (FAST-LIO, Point-LIO)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self._publisher = self._node.create_publisher(ImuMsg, topic_name, qos_profile)
        
        # Start publishing thread
        self._running = True
        self._thread = threading.Thread(target=self._publish_thread, daemon=True)
        self._thread.start()
        
        print(f"[INFO] HighFreqImuPublisher initialized on topic: {topic_name}")
        print(f"[INFO] Publish rate: {publish_rate} Hz, Frame ID: {frame_id}")
    
    def update_imu_data(self, ang_vel: np.ndarray, lin_vel: np.ndarray, 
                        orientation: np.ndarray, gravity: np.ndarray = None,
                        sim_time: float = None):
        """
        Update IMU data from robot state.
        
        This should be called from the simulation loop at each physics step.
        
        Args:
            ang_vel: Angular velocity in body frame (3,) in rad/s
            lin_vel: Linear velocity in body frame (3,) in m/s
            orientation: Orientation quaternion (4,) as (w, x, y, z) - Isaac Sim convention
            gravity: Projected gravity vector in body frame (3,), if None, uses [0, 0, -9.81]
            sim_time: Current simulation time in seconds (from Isaac Sim timeline)
                      This is critical for synchronization with LiDAR timestamps.
        """
        with self._lock:
            # Update simulation time
            if sim_time is not None:
                self._sim_time = sim_time
                self._sim_time_updated = True
            
            # Store angular velocity directly
            self._ang_vel = ang_vel.copy() if isinstance(ang_vel, np.ndarray) else ang_vel.cpu().numpy().flatten()
            
            # Convert linear velocity to numpy
            if not isinstance(lin_vel, np.ndarray):
                lin_vel = lin_vel.cpu().numpy().flatten()
            
            # Compute linear acceleration by numerical differentiation using simulation time
            dt = self._sim_time - self._prev_sim_time
            if dt > 0.0001:  # Avoid division by zero
                self._lin_acc = (lin_vel - self._prev_lin_vel) / dt
                # Add gravity effect (IMU measures acceleration including gravity)
                if gravity is not None:
                    if not isinstance(gravity, np.ndarray):
                        gravity = gravity.cpu().numpy().flatten()
                    # Subtract gravity to get proper acceleration (sensor measures a = measured - g)
                    self._lin_acc = self._lin_acc - gravity
                else:
                    # Default gravity in world Z-down
                    self._lin_acc[2] += 9.81
            
            self._prev_lin_vel = lin_vel.copy()
            self._prev_sim_time = self._sim_time
            
            # Convert orientation from Isaac Sim (w, x, y, z) to ROS (x, y, z, w)
            if not isinstance(orientation, np.ndarray):
                orientation = orientation.cpu().numpy().flatten()
            # Isaac Sim uses (w, x, y, z), ROS uses (x, y, z, w)
            self._orientation = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
    
    def _publish_thread(self):
        """Thread function to publish IMU data at high frequency."""
        while self._running and rclpy.ok():
            start_time = time_module.time()
            
            with self._lock:
                # Only publish if we have new data (simulation time updated)
                if not self._sim_time_updated:
                    # No new data, sleep briefly and continue
                    time_module.sleep(0.0001)
                    continue
                
                # Check if this is new data (avoid publishing duplicate timestamps)
                current_sim_time = self._sim_time
                if current_sim_time <= self._last_published_sim_time:
                    time_module.sleep(0.0001)
                    continue
                
                # Create and publish IMU message
                msg = ImuMsg()
                
                # Set header using SIMULATION TIME (critical for LiDAR synchronization)
                # Convert simulation time (float seconds) to ROS2 Time message
                sec = int(current_sim_time)
                nanosec = int((current_sim_time - sec) * 1e9)
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nanosec
                msg.header.frame_id = self.frame_id
                
                # Set orientation (x, y, z, w)
                msg.orientation.x = float(self._orientation[0])
                msg.orientation.y = float(self._orientation[1])
                msg.orientation.z = float(self._orientation[2])
                msg.orientation.w = float(self._orientation[3])
                
                # Set angular velocity
                msg.angular_velocity.x = float(self._ang_vel[0])
                msg.angular_velocity.y = float(self._ang_vel[1])
                msg.angular_velocity.z = float(self._ang_vel[2])
                
                # Set linear acceleration
                msg.linear_acceleration.x = float(self._lin_acc[0])
                msg.linear_acceleration.y = float(self._lin_acc[1])
                msg.linear_acceleration.z = float(self._lin_acc[2])
                
                # Mark data as consumed
                self._sim_time_updated = False
                self._last_published_sim_time = current_sim_time
            
            # Set covariance (unknown = -1 in first element, or use small values)
            # Using small covariance values for better SLAM integration
            msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            
            self._publisher.publish(msg)
            
            # Sleep to maintain target rate
            elapsed = time_module.time() - start_time
            sleep_time = self.publish_period - elapsed
            if sleep_time > 0:
                time_module.sleep(sleep_time)
    
    def shutdown(self):
        """Shutdown the publisher and cleanup resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self._node:
            self._node.destroy_node()
        
        print("[INFO] HighFreqImuPublisher shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass


class HighFreqLidarPublisher:
    """
    High-frequency LiDAR point cloud publisher for ROS2.
    
    This class publishes LiDAR point cloud data at a configurable frequency,
    using simulation time to ensure synchronization with IMU data.
    This is needed for SLAM algorithms like FAST-LIO and Point-LIO which require
    synchronized IMU and LiDAR data.
    
    The publisher runs in a separate thread and uses the same time source
    as the IMU publisher for proper sensor fusion.
    """
    
    def __init__(self, topic_name: str = "/point_cloud", 
                 frame_id: str = "lidar_frame",
                 publish_rate: float = 60.0,
                 domain_id: int = 0):
        """
        Initialize the high-frequency LiDAR publisher.
        
        Args:
            topic_name: ROS2 topic name for point cloud data
            frame_id: Frame ID for LiDAR messages
            publish_rate: Target publish rate in Hz
            domain_id: ROS2 domain ID
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not available. Cannot create HighFreqLidarPublisher.")
        
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.publish_rate = publish_rate
        self.publish_period = 1.0 / publish_rate
        
        # Point cloud data storage (thread-safe)
        self._lock = threading.Lock()
        self._points = None  # Point cloud data as numpy array (N, 3) or (N, 4) with intensity
        self._intensities = None  # Optional intensity data
        
        # Simulation time tracking (for synchronization with IMU)
        self._sim_time = 0.0
        self._sim_time_updated = False
        self._last_published_sim_time = -1.0
        
        # Set ROS_DOMAIN_ID if not already set
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain_id))
        
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS2 node and publisher
        self._node = rclpy.create_node('isaacsim_lidar_publisher')
        
        # Use reliable QoS for compatibility with SLAM algorithms
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self._publisher = self._node.create_publisher(PointCloud2, topic_name, qos_profile)
        
        # Start publishing thread
        self._running = True
        self._thread = threading.Thread(target=self._publish_thread, daemon=True)
        self._thread.start()
        
        print(f"[INFO] HighFreqLidarPublisher initialized on topic: {topic_name}")
        print(f"[INFO] Publish rate: {publish_rate} Hz, Frame ID: {frame_id}")
    
    def update_lidar_data(self, points: np.ndarray, intensities: np.ndarray = None,
                          sim_time: float = None):
        """
        Update LiDAR point cloud data.
        
        This should be called from the simulation loop when new LiDAR data is available.
        
        Args:
            points: Point cloud data as numpy array (N, 3) containing x, y, z coordinates
            intensities: Optional intensity data as numpy array (N,)
            sim_time: Current simulation time in seconds (from Isaac Sim timeline)
        """
        with self._lock:
            if sim_time is not None:
                self._sim_time = sim_time
                self._sim_time_updated = True
            
            if points is not None:
                if not isinstance(points, np.ndarray):
                    points = points.cpu().numpy()
                self._points = points.astype(np.float32)
            
            if intensities is not None:
                if not isinstance(intensities, np.ndarray):
                    intensities = intensities.cpu().numpy()
                self._intensities = intensities.astype(np.float32)
    
    def _create_pointcloud2_msg(self, points: np.ndarray, intensities: np.ndarray = None,
                                 sim_time: float = 0.0) -> PointCloud2:
        """
        Create a PointCloud2 message from numpy arrays.
        
        Args:
            points: Point cloud data (N, 3)
            intensities: Optional intensity data (N,)
            sim_time: Simulation time for the message timestamp
        
        Returns:
            PointCloud2 message
        """
        msg = PointCloud2()
        
        # Set header with simulation time
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = self.frame_id
        
        # Define point fields
        if intensities is not None:
            # XYZI format
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            point_step = 16
        else:
            # XYZ format
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            point_step = 12
        
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.height = 1
        msg.width = len(points)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Pack point data
        if intensities is not None:
            # Combine points and intensities
            data = np.zeros((len(points), 4), dtype=np.float32)
            data[:, :3] = points
            data[:, 3] = intensities
        else:
            data = points.astype(np.float32)
        
        msg.data = data.tobytes()
        
        return msg
    
    def _publish_thread(self):
        """Thread function to publish LiDAR data at specified frequency."""
        while self._running and rclpy.ok():
            start_time = time_module.time()
            
            with self._lock:
                # Only publish if we have new data
                if not self._sim_time_updated or self._points is None:
                    time_module.sleep(0.0001)
                    continue
                
                # Check if this is new data
                current_sim_time = self._sim_time
                if current_sim_time <= self._last_published_sim_time:
                    time_module.sleep(0.0001)
                    continue
                
                # Create and publish PointCloud2 message
                msg = self._create_pointcloud2_msg(
                    self._points, 
                    self._intensities,
                    current_sim_time
                )
                
                # Mark data as consumed
                self._sim_time_updated = False
                self._last_published_sim_time = current_sim_time
            
            self._publisher.publish(msg)
            
            # Sleep to maintain target rate
            elapsed = time_module.time() - start_time
            sleep_time = self.publish_period - elapsed
            if sleep_time > 0:
                time_module.sleep(sleep_time)
    
    def shutdown(self):
        """Shutdown the publisher and cleanup resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self._node:
            self._node.destroy_node()
        
        print("[INFO] HighFreqLidarPublisher shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass


class ClockPublisher:
    """
    ROS2 Clock publisher for simulation time.
    
    This class publishes the simulation time to /clock topic, which is
    essential for ROS2 nodes that use simulation time (use_sim_time:=true).
    Navigation stacks like Nav2 require synchronized time for proper operation.
    
    The publisher runs in a separate thread and uses Isaac Sim's timeline
    to get the current simulation time.
    """
    
    def __init__(self, topic_name: str = "/clock", 
                 publish_rate: float = 100.0,
                 domain_id: int = 0):
        """
        Initialize the clock publisher.
        
        Args:
            topic_name: ROS2 topic name for clock (usually /clock)
            publish_rate: Target publish rate in Hz
            domain_id: ROS2 domain ID
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not available. Cannot create ClockPublisher.")
        
        self.topic_name = topic_name
        self.publish_rate = publish_rate
        self.publish_period = 1.0 / publish_rate
        
        # Simulation time storage (thread-safe)
        self._lock = threading.Lock()
        self._sim_time = 0.0
        self._sim_time_updated = False
        self._last_published_sim_time = -1.0
        
        # Set ROS_DOMAIN_ID if not already set
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain_id))
        
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS2 node and publisher
        self._node = rclpy.create_node('isaacsim_clock_publisher')
        
        # Use best effort QoS for clock (standard for /clock topic)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self._publisher = self._node.create_publisher(Clock, topic_name, qos_profile)
        
        # Start publishing thread
        self._running = True
        self._thread = threading.Thread(target=self._publish_thread, daemon=True)
        self._thread.start()
        
        print(f"[INFO] ClockPublisher initialized on topic: {topic_name}")
        print(f"[INFO] Publish rate: {publish_rate} Hz")
    
    def update_sim_time(self, sim_time: float):
        """
        Update simulation time.
        
        This should be called from the simulation loop at each physics step.
        
        Args:
            sim_time: Current simulation time in seconds
        """
        with self._lock:
            self._sim_time = sim_time
            self._sim_time_updated = True
    
    def _publish_thread(self):
        """Thread function to publish clock at specified frequency."""
        while self._running and rclpy.ok():
            start_time = time_module.time()
            
            with self._lock:
                # Only publish if we have new data
                if not self._sim_time_updated:
                    time_module.sleep(0.001)
                    continue
                
                # Check if this is new data
                current_sim_time = self._sim_time
                if current_sim_time <= self._last_published_sim_time:
                    time_module.sleep(0.001)
                    continue
                
                # Create Clock message
                clock_msg = Clock()
                
                # Set time from simulation
                sec = int(current_sim_time)
                nanosec = int((current_sim_time - sec) * 1e9)
                clock_msg.clock.sec = sec
                clock_msg.clock.nanosec = nanosec
                
                # Mark data as consumed
                self._sim_time_updated = False
                self._last_published_sim_time = current_sim_time
            
            # Publish Clock message
            self._publisher.publish(clock_msg)
            
            # Sleep to maintain target rate
            elapsed = time_module.time() - start_time
            sleep_time = self.publish_period - elapsed
            if sleep_time > 0:
                time_module.sleep(sleep_time)
    
    def shutdown(self):
        """Shutdown the publisher and cleanup resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self._node:
            self._node.destroy_node()
        
        print("[INFO] ClockPublisher shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except:
            pass


class OdomTFPublisher:
    """
    Dynamic odom->base_link TF publisher for ROS2.
    
    This class publishes the transform from odom to base_link based on
    the robot's actual position and orientation in the simulation.
    This is essential for navigation stacks like Nav2 that require
    odom->base_link transforms for localization.
    
    The publisher runs in a separate thread and uses simulation time
    to ensure synchronization with other sensor data.
    """
    
    def __init__(self, topic_name: str = "/tf", 
                 odom_frame_id: str = "odom",
                 base_frame_id: str = "base_link",
                 publish_rate: float = 60.0,
                 domain_id: int = 0):
        """
        Initialize the odom TF publisher.
        
        Args:
            topic_name: ROS2 topic name for TF (usually /tf)
            odom_frame_id: Frame ID for the odom frame (parent)
            base_frame_id: Frame ID for the base_link frame (child)
            publish_rate: Target publish rate in Hz
            domain_id: ROS2 domain ID
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not available. Cannot create OdomTFPublisher.")
        
        # Import TF2 message types
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        
        self.topic_name = topic_name
        self.odom_frame_id = odom_frame_id
        self.base_frame_id = base_frame_id
        self.publish_rate = publish_rate
        self.publish_period = 1.0 / publish_rate
        
        # Robot pose storage (thread-safe)
        self._lock = threading.Lock()
        self._position = np.zeros(3)  # Position (x, y, z) in world/odom frame
        self._orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w) ROS convention
        
        # Simulation time tracking
        self._sim_time = 0.0
        self._sim_time_updated = False
        self._last_published_sim_time = -1.0
        
        # Initial pose offset (to make odom start at origin)
        self._initial_position = None
        self._initial_orientation_inv = None  # Inverse of initial orientation for proper 3D transform
        
        # Set ROS_DOMAIN_ID if not already set
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain_id))
        
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS2 node and publisher
        self._node = rclpy.create_node('isaacsim_odom_tf_publisher')
        
        # Use reliable QoS for TF
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self._publisher = self._node.create_publisher(TFMessage, topic_name, qos_profile)
        self._TransformStamped = TransformStamped
        self._TFMessage = TFMessage
        
        # Start publishing thread
        self._running = True
        self._thread = threading.Thread(target=self._publish_thread, daemon=True)
        self._thread.start()
        
        print(f"[INFO] OdomTFPublisher initialized on topic: {topic_name}")
        print(f"[INFO] Publishing TF: {odom_frame_id} -> {base_frame_id}")
        print(f"[INFO] Publish rate: {publish_rate} Hz")
    
    @staticmethod
    def _quat_conjugate(q):
        """Compute quaternion conjugate (inverse for unit quaternion). Input/output: (w, x, y, z)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def _quat_multiply(q1, q2):
        """Multiply two quaternions. Input/output: (w, x, y, z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def _quat_rotate_vector(q, v):
        """Rotate vector v by quaternion q. q: (w, x, y, z), v: (x, y, z)."""
        # Convert vector to quaternion form (0, x, y, z)
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        q_conj = OdomTFPublisher._quat_conjugate(q)
        # Rotated vector = q * v * q^-1
        result = OdomTFPublisher._quat_multiply(
            OdomTFPublisher._quat_multiply(q, v_quat), q_conj
        )
        return result[1:4]  # Return (x, y, z) part

    def update_robot_pose(self, position: np.ndarray, orientation: np.ndarray,
                          sim_time: float = None):
        """
        Update robot pose from simulation.
        
        This should be called from the simulation loop at each physics step.
        
        Args:
            position: Robot position in world frame (3,) as (x, y, z)
            orientation: Orientation quaternion (4,) as (w, x, y, z) - Isaac Sim convention
            sim_time: Current simulation time in seconds
        """
        with self._lock:
            if sim_time is not None:
                self._sim_time = sim_time
                self._sim_time_updated = True
            
            # Convert position to numpy
            if not isinstance(position, np.ndarray):
                position = position.cpu().numpy().flatten()
            
            # Convert orientation to numpy (w, x, y, z)
            if not isinstance(orientation, np.ndarray):
                orientation = orientation.cpu().numpy().flatten()
            
            # Set initial pose on first update (to make odom start at origin)
            if self._initial_position is None:
                self._initial_position = position.copy()
                # Store inverse of initial orientation for proper 3D transform
                self._initial_orientation_inv = self._quat_conjugate(orientation)
                print(f"[INFO] OdomTFPublisher: Initial pose set at position {self._initial_position}")
            
            # Compute relative position from initial position
            rel_position_world = position - self._initial_position
            
            # Rotate relative position into odom frame using inverse of initial orientation
            # This properly handles full 3D rotation, not just yaw
            self._position = self._quat_rotate_vector(self._initial_orientation_inv, rel_position_world)
            
            # Compute relative orientation: q_rel = q_init^-1 * q_current
            # This gives the rotation from initial orientation to current orientation
            rel_orientation_wxyz = self._quat_multiply(self._initial_orientation_inv, orientation)
            
            # Convert from Isaac Sim (w, x, y, z) to ROS (x, y, z, w)
            self._orientation[0] = rel_orientation_wxyz[1]  # x
            self._orientation[1] = rel_orientation_wxyz[2]  # y
            self._orientation[2] = rel_orientation_wxyz[3]  # z
            self._orientation[3] = rel_orientation_wxyz[0]  # w
    
    def _publish_thread(self):
        """Thread function to publish odom TF at specified frequency."""
        while self._running and rclpy.ok():
            start_time = time_module.time()
            
            with self._lock:
                # Only publish if we have new data
                if not self._sim_time_updated:
                    time_module.sleep(0.001)
                    continue
                
                # Check if this is new data
                current_sim_time = self._sim_time
                if current_sim_time <= self._last_published_sim_time:
                    time_module.sleep(0.001)
                    continue
                
                # Create TransformStamped message
                t = self._TransformStamped()
                
                # Set header with simulation time
                sec = int(current_sim_time)
                nanosec = int((current_sim_time - sec) * 1e9)
                t.header.stamp.sec = sec
                t.header.stamp.nanosec = nanosec
                t.header.frame_id = self.odom_frame_id
                t.child_frame_id = self.base_frame_id
                
                # Set translation
                t.transform.translation.x = float(self._position[0])
                t.transform.translation.y = float(self._position[1])
                t.transform.translation.z = float(self._position[2])
                
                # Set rotation (quaternion x, y, z, w)
                t.transform.rotation.x = float(self._orientation[0])
                t.transform.rotation.y = float(self._orientation[1])
                t.transform.rotation.z = float(self._orientation[2])
                t.transform.rotation.w = float(self._orientation[3])
                
                # Mark data as consumed
                self._sim_time_updated = False
                self._last_published_sim_time = current_sim_time
            
            # Publish TFMessage
            tf_msg = self._TFMessage()
            tf_msg.transforms.append(t)
            self._publisher.publish(tf_msg)
            
            # Sleep to maintain target rate
            elapsed = time_module.time() - start_time
            sleep_time = self.publish_period - elapsed
            if sleep_time > 0:
                time_module.sleep(sleep_time)
    
    def shutdown(self):
        """Shutdown the publisher and cleanup resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self._node:
            self._node.destroy_node()
        
        print("[INFO] OdomTFPublisher shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass


# Enable required extensions for ROS2 camera publishing
def enable_required_extensions():
    """Enable all required extensions for ROS2 camera publishing."""
    import omni.kit.app
    
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    
    # All required extensions for ROS2 camera publishing
    required_extensions = [
        # ROS2 bridge extensions (try new name first, then legacy)
        ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"),
        # Core nodes extensions (try new name first, then legacy)
        ("isaacsim.core.nodes", "omni.isaac.core_nodes"),
    ]
    
    enabled_extensions = []
    
    for extension_pair in required_extensions:
        enabled = False
        for ext_name in extension_pair:
            if extension_manager.is_extension_enabled(ext_name):
                print(f"[INFO] Extension '{ext_name}' is already enabled")
                enabled_extensions.append(ext_name)
                enabled = True
                break
            else:
                try:
                    extension_manager.set_extension_enabled_immediate(ext_name, True)
                    print(f"[INFO] Enabled extension: {ext_name}")
                    enabled_extensions.append(ext_name)
                    enabled = True
                    break
                except Exception as e:
                    print(f"[DEBUG] Could not enable {ext_name}: {e}")
                    continue
        
        if not enabled:
            print(f"[WARN] Could not enable any extension from: {extension_pair}")
    
    return len(enabled_extensions) > 0

# Try to enable required extensions
extensions_enabled = enable_required_extensions()
simulation_app.update()  # Update to ensure extensions are loaded


def remove_usd_robots(stage):
    """
    Remove all robots from the USD stage (except those under /World/envs).
    
    This is useful when the USD file contains pre-existing robot models
    that you want to remove before spawning your own robot.
    
    Args:
        stage: The USD stage object
    """
    
    # List of common robot prim paths to remove
    robot_paths_to_check = [
        "/World/walkers1",
    ]
    
    removed_count = 0
    for prim_path in robot_paths_to_check:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            stage.RemovePrim(prim_path)
            print(f"[INFO] Removed prim at {prim_path}")
            removed_count += 1
    
    if removed_count == 0:
        print("[INFO] No pre-existing robots found in USD scene")


def create_camera_on_robot(stage, robot_prim_path: str, camera_name: str = "head_camera",
                           local_position: tuple = (0.3, 0.0, 0.3),
                           local_rotation: tuple = (0.0, 0.0, 0.0),
                           width: int = 640, height: int = 480):
    """
    Create a camera attached to the robot.
    
    Args:
        stage: USD stage
        robot_prim_path: Path to the robot prim
        camera_name: Name for the camera
        local_position: Local position offset from parent (x, y, z) in meters
        local_rotation: Local rotation offset from parent (roll, pitch, yaw) in degrees
        width: Camera image width
        height: Camera image height
    
    Returns:
        str: Path to the created camera prim
    """
    # Find the robot's base/pelvis link to attach camera
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"[WARN] Robot prim not found at {robot_prim_path}")
        return None
    
    # Find a suitable parent body (pelvis or base_link)
    possible_parents = ["pelvis", "base_link", "base", "torso", "chassis"]
    parent_path = None
    
    for parent_name in possible_parents:
        test_path = f"{robot_prim_path}/{parent_name}"
        if stage.GetPrimAtPath(test_path).IsValid():
            parent_path = test_path
            break
    
    if parent_path is None:
        # If no specific body found, attach directly to robot root
        parent_path = robot_prim_path
        print(f"[INFO] No standard body found, attaching camera to robot root: {parent_path}")
    else:
        print(f"[INFO] Attaching camera to: {parent_path}")
    
    camera_path = f"{parent_path}/{camera_name}"
    
    # Create the camera prim
    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    
    # Set camera attributes
    camera_prim.GetHorizontalApertureAttr().Set(20.955)  # Standard 35mm equivalent
    camera_prim.GetVerticalApertureAttr().Set(15.2908)
    camera_prim.GetFocalLengthAttr().Set(24.0)
    camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
    
    # Set local transform
    xform = UsdGeom.Xformable(camera_prim.GetPrim())
    
    # Create translation operation
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(local_position[0], local_position[1], local_position[2]))
    
    # Create rotation operations (XYZ Euler)
    if any(r != 0 for r in local_rotation):
        rotate_x_op = xform.AddRotateXOp()
        rotate_x_op.Set(local_rotation[0])
        rotate_y_op = xform.AddRotateYOp()
        rotate_y_op.Set(local_rotation[1])
        rotate_z_op = xform.AddRotateZOp()
        rotate_z_op.Set(local_rotation[2])
    
    print(f"[INFO] Created camera at: {camera_path}")
    return camera_path


def create_rtx_lidar_on_robot(stage, robot_prim_path: str, lidar_name: str = "mid360_lidar",
                              local_position: tuple = (0.0, 0.0, 0.4),
                              local_rotation: tuple = (0.0, 0.0, 0.0)):
    """
    Create an RTX LiDAR sensor attached to the robot using Isaac Sim's built-in Example_Rotary config.
    
    This creates a 360° rotary LiDAR suitable for SLAM and navigation applications.
    
    Args:
        stage: USD stage
        robot_prim_path: Path to the robot prim
        lidar_name: Name for the lidar sensor
        local_position: Local position offset from parent (x, y, z) in meters
        local_rotation: Local rotation offset from parent (roll, pitch, yaw) in degrees
    
    Returns:
        str: Path to the created lidar prim, or None if creation failed
    """
    # Find the robot's base/pelvis link to attach lidar
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"[WARN] Robot prim not found at {robot_prim_path}")
        return None
    
    # Find a suitable parent body (pelvis or base_link)
    possible_parents = ["pelvis", "base_link", "base", "torso", "chassis"]
    parent_path = None
    
    for parent_name in possible_parents:
        test_path = f"{robot_prim_path}/{parent_name}"
        if stage.GetPrimAtPath(test_path).IsValid():
            parent_path = test_path
            break
    
    if parent_path is None:
        # If no specific body found, attach directly to robot root
        parent_path = robot_prim_path
        print(f"[INFO] No standard body found, attaching lidar to robot root: {parent_path}")
    else:
        print(f"[INFO] Attaching lidar to: {parent_path}")
    
    lidar_path = f"{parent_path}/{lidar_name}"
    
    # Calculate orientation from local rotation (Euler angles to quaternion)
    import math
    roll = math.radians(local_rotation[0])
    pitch = math.radians(local_rotation[1])
    yaw = math.radians(local_rotation[2])
    
    # Euler to quaternion conversion (ZYX order)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    try:
        # Execute the command to create the RTX LiDAR with Example_Rotary config
        # This provides a 360° rotary LiDAR suitable for SLAM and navigation
        print(f"[INFO] Creating RTX LiDAR with config: Example_Rotary")
        
        success, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=lidar_name,
            parent=parent_path,
            config="Example_Rotary",  # Built-in 360° rotary LiDAR config
            translation=Gf.Vec3d(local_position[0], local_position[1], local_position[2]),
            orientation=Gf.Quatd(qw, qx, qy, qz),
        )
        
        if success:
            print(f"[INFO] Created RTX LiDAR at: {lidar_path}")
            return lidar_path
        else:
            print(f"[ERROR] Failed to create RTX LiDAR")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to create RTX LiDAR: {e}")
        return None


def setup_ros2_lidar_graph(lidar_prim_path: str, point_cloud_topic: str, 
                           frame_id: str, domain_id: int = 0):
    """
    Setup OmniGraph for publishing RTX LiDAR data to ROS2 topics.
    
    Args:
        lidar_prim_path: Path to the lidar prim
        point_cloud_topic: ROS2 topic name for point cloud
        frame_id: Frame ID for the lidar
        domain_id: ROS2 domain ID
    
    Returns:
        og.Graph: The created OmniGraph
    """
    
    graph_path = "/World/ROS2_Lidar_Graph"
    
    keys = og.Controller.Keys
    
    # Delete existing graph if it exists
    try:
        existing_graph = og.get_graph_by_path(graph_path)
        if existing_graph is not None and existing_graph.is_valid():
            print(f"[DEBUG] Deleting existing lidar graph at {graph_path}")
            og.Controller.delete_graph(graph_path)
    except Exception as e:
        print(f"[DEBUG] No existing lidar graph to delete: {e}")
    
    # Try different node type naming conventions (new vs legacy)
    node_type_variants = [
        {
            "prefix": "isaacsim",
            "context": "isaacsim.ros2.bridge.ROS2Context",
            "lidar_helper": "isaacsim.ros2.bridge.ROS2RtxLidarHelper",
            "create_render_product": "isaacsim.core.nodes.IsaacCreateRenderProduct",
        },
        {
            "prefix": "omni.isaac",
            "context": "omni.isaac.ros2_bridge.ROS2Context",
            "lidar_helper": "omni.isaac.ros2_bridge.ROS2RtxLidarHelper",
            "create_render_product": "omni.isaac.core_nodes.IsaacCreateRenderProduct",
        },
    ]
    
    last_error = None
    
    for variant in node_type_variants:
        # Clean up any partially created graph before each attempt
        try:
            og.Controller.delete_graph(graph_path)
        except:
            pass
        
        try:
            print(f"[DEBUG] Trying lidar node types: {variant['context']}")
            
            # Create the action graph
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2Context", variant["context"]),
                        ("CreateRenderProduct", variant["create_render_product"]),
                        ("ROS2LidarHelper", variant["lidar_helper"]),
                    ],
                    keys.SET_VALUES: [
                        # ROS2 Context settings
                        ("ROS2Context.inputs:domain_id", domain_id),
                        ("ROS2Context.inputs:useDomainIDEnvVar", False),
                        
                        # Render Product settings - use lidar prim as camera prim
                        ("CreateRenderProduct.inputs:cameraPrim", lidar_prim_path),
                        ("CreateRenderProduct.inputs:enabled", True),
                        
                        # Lidar Helper settings for PointCloud2
                        ("ROS2LidarHelper.inputs:type", "point_cloud"),
                        ("ROS2LidarHelper.inputs:topicName", point_cloud_topic),
                        ("ROS2LidarHelper.inputs:frameId", frame_id),
                        ("ROS2LidarHelper.inputs:fullScan", False),  # Publish after full scan
                    ],
                    keys.CONNECT: [
                        # Connect tick to render product creation
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        
                        # Connect render product to lidar helper
                        ("CreateRenderProduct.outputs:execOut", "ROS2LidarHelper.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "ROS2LidarHelper.inputs:renderProductPath"),
                        
                        # Connect ROS2 context
                        ("ROS2Context.outputs:context", "ROS2LidarHelper.inputs:context"),
                    ],
                },
            )
            
            print(f"[INFO] Created ROS2 LiDAR graph at: {graph_path}")
            print(f"[INFO] Point cloud topic: {point_cloud_topic}")
            
            return graph
            
        except Exception as e:
            last_error = e
            print(f"[DEBUG] Failed with lidar node types {variant['context']}: {e}")
            # Try to clean up the partially created graph
            try:
                og.Controller.delete_graph(graph_path)
            except:
                pass
            continue
    
    # If all variants failed, raise the last error
    raise RuntimeError(f"Failed to create ROS2 LiDAR graph with any node type variant. Last error: {last_error}")


def setup_ros2_camera_graph(camera_prim_path: str, rgb_topic: str, depth_topic: str, 
                            camera_info_topic: str, frame_id: str, domain_id: int = 0):
    """
    Setup OmniGraph for publishing camera data to ROS2 topics.
    
    Args:
        camera_prim_path: Path to the camera prim
        rgb_topic: ROS2 topic name for RGB image
        depth_topic: ROS2 topic name for depth image
        camera_info_topic: ROS2 topic name for camera info
        frame_id: Frame ID for the camera
        domain_id: ROS2 domain ID
    
    Returns:
        og.Graph: The created OmniGraph
    """
    
    graph_path = "/World/ROS2_Camera_Graph"
    
    keys = og.Controller.Keys
    
    # Delete existing graph if it exists
    try:
        existing_graph = og.get_graph_by_path(graph_path)
        if existing_graph is not None and existing_graph.is_valid():
            print(f"[DEBUG] Deleting existing graph at {graph_path}")
            og.Controller.delete_graph(graph_path)
    except Exception as e:
        print(f"[DEBUG] No existing graph to delete: {e}")
    
    # Try different node type naming conventions (new vs legacy)
    node_type_variants = [
        {
            "prefix": "isaacsim",
            "context": "isaacsim.ros2.bridge.ROS2Context",
            "camera_helper": "isaacsim.ros2.bridge.ROS2CameraHelper",
            "camera_info": "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
            "create_render_product": "isaacsim.core.nodes.IsaacCreateRenderProduct",
        },
        {
            "prefix": "omni.isaac",
            "context": "omni.isaac.ros2_bridge.ROS2Context",
            "camera_helper": "omni.isaac.ros2_bridge.ROS2CameraHelper",
            "camera_info": "omni.isaac.ros2_bridge.ROS2CameraInfoHelper",
            "create_render_product": "omni.isaac.core_nodes.IsaacCreateRenderProduct",
        },
    ]
    
    last_error = None
    
    for variant in node_type_variants:
        # Clean up any partially created graph before each attempt
        try:
            og.Controller.delete_graph(graph_path)
        except:
            pass
        
        try:
            print(f"[DEBUG] Trying node types: {variant['context']}")
            
            # Create the action graph
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2Context", variant["context"]),
                        ("CreateRenderProduct", variant["create_render_product"]),
                        ("ROS2CameraHelperRGB", variant["camera_helper"]),
                        ("ROS2CameraHelperDepth", variant["camera_helper"]),
                        ("ROS2CameraInfoHelper", variant["camera_info"]),
                    ],
                    keys.SET_VALUES: [
                        # ROS2 Context settings
                        ("ROS2Context.inputs:domain_id", domain_id),
                        ("ROS2Context.inputs:useDomainIDEnvVar", False),
                        
                        # Render Product settings
                        ("CreateRenderProduct.inputs:cameraPrim", camera_prim_path),
                        ("CreateRenderProduct.inputs:enabled", True),
                        ("CreateRenderProduct.inputs:width", args_cli.camera_width),
                        ("CreateRenderProduct.inputs:height", args_cli.camera_height),
                        
                        # RGB Camera Helper settings
                        ("ROS2CameraHelperRGB.inputs:type", "rgb"),
                        ("ROS2CameraHelperRGB.inputs:topicName", rgb_topic),
                        ("ROS2CameraHelperRGB.inputs:frameId", frame_id),
                        ("ROS2CameraHelperRGB.inputs:enableSemanticLabels", False),
                        
                        # Depth Camera Helper settings
                        ("ROS2CameraHelperDepth.inputs:type", "depth"),
                        ("ROS2CameraHelperDepth.inputs:topicName", depth_topic),
                        ("ROS2CameraHelperDepth.inputs:frameId", frame_id),
                        
                        # Camera Info Helper settings
                        ("ROS2CameraInfoHelper.inputs:topicName", camera_info_topic),
                        ("ROS2CameraInfoHelper.inputs:frameId", frame_id),
                    ],
                    keys.CONNECT: [
                        # Connect tick directly to render product creation
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        
                        # Connect render product to camera helpers
                        ("CreateRenderProduct.outputs:execOut", "ROS2CameraHelperRGB.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperRGB.inputs:renderProductPath"),
                        
                        ("CreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepth.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepth.inputs:renderProductPath"),
                        
                        ("CreateRenderProduct.outputs:execOut", "ROS2CameraInfoHelper.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraInfoHelper.inputs:renderProductPath"),
                        
                        # Connect ROS2 context
                        ("ROS2Context.outputs:context", "ROS2CameraHelperRGB.inputs:context"),
                        ("ROS2Context.outputs:context", "ROS2CameraHelperDepth.inputs:context"),
                        ("ROS2Context.outputs:context", "ROS2CameraInfoHelper.inputs:context"),
                    ],
                },
            )
            
            print(f"[INFO] Created ROS2 camera graph at: {graph_path}")
            print(f"[INFO] RGB topic: {rgb_topic}")
            print(f"[INFO] Depth topic: {depth_topic}")
            print(f"[INFO] Camera info topic: {camera_info_topic}")
            
            return graph
            
        except Exception as e:
            last_error = e
            print(f"[DEBUG] Failed with node types {variant['context']}: {e}")
            # Try to clean up the partially created graph
            try:
                og.Controller.delete_graph(graph_path)
            except:
                pass
            continue
    
    # If all variants failed, raise the last error
    raise RuntimeError(f"Failed to create ROS2 camera graph with any node type variant. Last error: {last_error}")


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
        policy_path = os.path.abspath(args_cli.policy_path)
        file_content = omni.client.read_file(policy_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        policy = torch.jit.load(file, map_location=device)
        print(f"[INFO] Loaded policy from: {policy_path}")

        # get environment configuration
        env_class_name = args_cli.task
        env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

        # modify configuration for USD environment inference
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.events.push_robot = None
        env_cfg.scene.max_episode_length_s = 1000.0  # Long episode for demo
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.scene.env_spacing = 2.5
        env_cfg.commands.rel_standing_envs = 0.0

        # set terrain to USD (default: ../sense/museum/museum.usd)
        usd_path = os.path.abspath(args_cli.usd_path)
        print(f"[INFO] Using USD environment: {usd_path}")

        # override terrain configuration for USD environment
        env_cfg.scene.terrain_type = "usd"
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.usd_path = usd_path

        # disable height scanner for USD environment (mesh not available)
        env_cfg.scene.height_scanner.enable_height_scan = False

        if args_cli.seed is not None:
            env_cfg.scene.seed = args_cli.seed

        # set device
        env_cfg.device = device

        # IMPORTANT: Remove robots from USD BEFORE creating environment
        # This prevents PhysX from creating tensor views for prims that will be deleted
        stage = Usd.Stage.Open(usd_path)
        remove_usd_robots(stage)
        # Save the modified USD to a temporary file
        import tempfile
        temp_usd = tempfile.NamedTemporaryFile(suffix=".usd", delete=False)
        temp_usd_path = temp_usd.name
        temp_usd.close()
        stage.Export(temp_usd_path)
        print(f"[INFO] Saved cleaned USD to temporary file: {temp_usd_path}")
        
        # Update config to use the cleaned USD
        env_cfg.scene.usd_path = temp_usd_path

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
        
        # Create RTX LiDAR on the robot if enabled
        if args_cli.enable_lidar:
            lidar_path = create_rtx_lidar_on_robot(
                stage=current_stage,
                robot_prim_path=robot_prim_path,
                lidar_name="mid360_lidar",
                local_position=(0.0, 0.0, 1.0),  # 机器人 pelvis 上方 1.0m
                local_rotation=(0.0, 0.0, 0.0),
            )
            
            # Update simulation to initialize the lidar
            simulation_app.update()
            
            if lidar_path:
                # Setup ROS2 LiDAR publishing graph
                try:
                    ros2_lidar_graph = setup_ros2_lidar_graph(
                        lidar_prim_path=lidar_path,
                        point_cloud_topic=args_cli.lidar_topic,
                        frame_id=args_cli.lidar_frame_id,
                        domain_id=args_cli.ros2_domain_id
                    )
                    print("[INFO] ROS2 LiDAR publishing enabled successfully!")
                    print(f"[INFO] LiDAR Point Cloud topic: {args_cli.lidar_topic}")
                    print(f"[INFO] To view point cloud: ros2 topic echo {args_cli.lidar_topic}")
                    print(f"[INFO] To visualize in RViz2: Add PointCloud2 display with topic {args_cli.lidar_topic}")
                except Exception as e:
                    print(f"[ERROR] Failed to setup ROS2 LiDAR graph: {e}")
                    print("[WARN] Continuing without ROS2 LiDAR publishing...")
            else:
                print("[WARN] LiDAR creation failed, skipping ROS2 LiDAR publishing setup")
        else:
            print("[INFO] LiDAR disabled. Use --enable_lidar to enable RTX LiDAR sensor.")

        # Setup ROS2 cmd_vel subscriber if enabled
        if args_cli.enable_cmd_vel:
            if ROS2_AVAILABLE:
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
                print("[WARN] rclpy not available. cmd_vel subscriber disabled.")
        else:
            print("[INFO] cmd_vel subscriber disabled. Use --enable_cmd_vel to enable velocity control via ROS2.")

        # Setup high-frequency IMU publisher if enabled
        if args_cli.enable_high_freq_imu:
            if ROS2_AVAILABLE:
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
                print("[WARN] rclpy not available. High-frequency IMU publisher disabled.")
        else:
            print("[INFO] High-frequency IMU publisher disabled. Use --enable_high_freq_imu to enable.")

        # Setup odom TF publisher if enabled
        if args_cli.enable_odom_tf:
            if ROS2_AVAILABLE:
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
                print("[WARN] rclpy not available. Odom TF publisher disabled.")
        else:
            print("[INFO] Odom TF publisher disabled. Use --enable_odom_tf to enable.")

        # Setup clock publisher if enabled
        if args_cli.enable_clock:
            if ROS2_AVAILABLE:
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
                print("[WARN] rclpy not available. Clock publisher disabled.")
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
        if ROS2_AVAILABLE:
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

# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
ROS2 publisher classes for simulation data.

Contains:
- HighFreqImuPublisher: High-frequency IMU data publisher
- HighFreqLidarPublisher: LiDAR point cloud publisher
- ClockPublisher: Simulation clock publisher
- OdomTFPublisher: Odom→base_link TF publisher
"""

import os
import threading
import time as time_module

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu as ImuMsg
from sensor_msgs.msg import PointCloud2, PointField
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


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
                t = TransformStamped()

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
            tf_msg = TFMessage()
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

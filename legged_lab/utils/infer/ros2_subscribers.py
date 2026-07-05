# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
ROS2 subscriber classes for robot control.

Contains:
- CmdVelSubscriber: Subscribes to geometry_msgs/Twist for velocity commands
"""

import os
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist


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

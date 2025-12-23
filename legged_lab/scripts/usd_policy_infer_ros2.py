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
    # Run with default warehouse USD environment
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/exported/policy.pt

    # Run with custom USD environment
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/exported/policy.pt --usd_path /path/to/custom.usd

    # Run with custom camera topic names
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/policy.pt --rgb_topic /camera/rgb --depth_topic /camera/depth

    # Run with RTX LiDAR enabled
    python legged_lab/scripts/usd_policy_infer_ros2.py --task walk --policy_path /path/to/policy.pt --enable_lidar --lidar_topic /point_cloud

ROS2 Topics Published:
    - /rgb (sensor_msgs/Image): RGB camera image
    - /depth (sensor_msgs/Image): Depth camera image
    - /camera_info (sensor_msgs/CameraInfo): Camera intrinsic parameters
    - /point_cloud (sensor_msgs/PointCloud2): RTX LiDAR point cloud data (when --enable_lidar is set)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# add argparse arguments
parser = argparse.ArgumentParser(description="Policy inference for TienKung robot in a USD environment with ROS2 camera publishing.")
parser.add_argument("--task", type=str, default="walk", help="Name of the task.")
parser.add_argument("--policy_path", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--usd_path", type=str, default=None, help="Path to custom USD environment file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# ROS2 camera configuration
parser.add_argument("--rgb_topic", type=str, default="/rgb", help="ROS2 topic name for RGB image.")
parser.add_argument("--depth_topic", type=str, default="/depth", help="ROS2 topic name for depth image.")
parser.add_argument("--camera_info_topic", type=str, default="/camera_info", help="ROS2 topic name for camera info.")
parser.add_argument("--camera_frame_id", type=str, default="robot_camera", help="Frame ID for camera messages.")
parser.add_argument("--camera_width", type=int, default=640, help="Camera image width.")
parser.add_argument("--camera_height", type=int, default=480, help="Camera image height.")
parser.add_argument("--ros2_domain_id", type=int, default=0, help="ROS2 domain ID.")
# RTX LiDAR configuration
parser.add_argument("--enable_lidar", action="store_true", help="Enable RTX LiDAR sensor.")
parser.add_argument("--lidar_topic", type=str, default="/point_cloud", help="ROS2 topic name for LiDAR point cloud.")
parser.add_argument("--lidar_frame_id", type=str, default="lidar_frame", help="Frame ID for LiDAR messages.")

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
from pxr import Usd, UsdGeom, Gf

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from legged_lab.envs import *  # noqa:F401, F403

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
    Create an RTX LiDAR sensor attached to the robot with custom attributes.
    
    Args:
        stage: USD stage
        robot_prim_path: Path to the robot prim
        lidar_name: Name for the lidar sensor
        local_position: Local position offset from parent (x, y, z) in meters
        local_rotation: Local rotation offset from parent (roll, pitch, yaw) in degrees
    
    Returns:
        str: Path to the created lidar prim
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
    
    # Custom sensor attributes for Mid-360 like LiDAR
    sensor_attributes = {
        'omni:sensor:Core:scanType': "ROTARY",
        'omni:sensor:Core:intensityProcessing': "NORMALIZATION",
        'omni:sensor:Core:rotationDirection': "CW",
        'omni:sensor:Core:rayType': "IDEALIZED",
        'omni:sensor:Core:nearRangeM': 0.1,
        'omni:sensor:Core:farRangeM': 40.0,
        'omni:sensor:Core:rangeResolutionM': 0.004,
        'omni:sensor:Core:rangeAccuracyM': 0.025,
        'omni:sensor:Core:avgPowerW': 0.002,
        'omni:sensor:Core:minReflectance': 0.1,
        'omni:sensor:Core:minReflectanceRange': 70.0,
        'omni:sensor:Core:wavelengthNm': 905.0,
        'omni:sensor:Core:pulseTimeNs': 6,
        'omni:sensor:Core:azimuthErrorMean': 0.1,
        'omni:sensor:Core:azimuthErrorStd': 0.5,
        'omni:sensor:Core:elevationErrorMean': 0.1,
        'omni:sensor:Core:elevationErrorStd': 0.5,
        'omni:sensor:Core:maxReturns': 2,
        'omni:sensor:Core:scanRateBaseHz': 20.0,
        'omni:sensor:Core:reportRateBaseHz': 7761,
        'omni:sensor:Core:numberOfEmitters': 40,
        'omni:sensor:Core:numberOfChannels': 40,
        'omni:sensor:Core:rangeOffset': 0.03,
        'omni:sensor:Core:intensityMappingType': "LINEAR",
        'omni:sensor:Core:emitterState:s001:azimuthDeg': [0] * 40,
        'omni:sensor:Core:emitterState:s001:elevationDeg': [
            -7.0, -5.525, -4.050, -2.575, -1.1004, 0.374, 1.849, 3.324, 4.799, 6.274,
            7.7494, 9.2249, 10.699, 12.174, 13.645, 15.1243, 16.5999, 18.074, 19.5499, 21.024,
            22.493, 23.9749, 25.44, 26.924, 28.39, 29.8743, 31.3499, 32.824, 34.29, 35.774,
            37.2486, 38.724, 40.19, 41.674, 43.14, 44.624, 46.09, 47.574, 49.048, 50.524
        ],
        'omni:sensor:Core:emitterState:s001:fireTimeNs': [i * 1000 for i in range(40)],
        'omni:sensor:Core:emitterState:s001:distanceCorrectionM': [0.0] * 40,
        'omni:sensor:Core:emitterState:s001:focalDistM': [0.0] * 40,
        'omni:sensor:Core:emitterState:s001:focalSlope': [0.0] * 40,
        'omni:sensor:Core:emitterState:s001:horOffsetM': [0.0] * 40,
        'omni:sensor:Core:emitterState:s001:reportRateDiv': [0.0] * 40,
        'omni:sensor:Core:emitterState:s001:vertOffsetM': [0.0] * 40,
        'omni:sensor:Core:emitterState:s001:channelId': list(range(1, 41)),
    }
    
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
        # Execute the command to create the RTX LiDAR
        success, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=lidar_name,
            parent=parent_path,
            config="Example_Rotary",  # Valid default rotating LiDAR config for Isaac Sim 4.5
            translation=Gf.Vec3d(local_position[0], local_position[1], local_position[2]),
            orientation=Gf.Quatd(qw, qx, qy, qz),
        )
        
        if success:
            print(f"[INFO] Created RTX LiDAR at: {lidar_path}")
            
            # Apply custom sensor attributes
            lidar_prim = stage.GetPrimAtPath(lidar_path)
            if lidar_prim.IsValid():
                for attr_name, attr_value in sensor_attributes.items():
                    # Skip array attributes that might not be supported directly
                    if isinstance(attr_value, list):
                        continue
                    try:
                        attr = lidar_prim.GetAttribute(attr_name)
                        if attr.IsValid():
                            attr.Set(attr_value)
                    except Exception as e:
                        print(f"[DEBUG] Could not set attribute {attr_name}: {e}")
            
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
                        ("ROS2LidarHelper.inputs:fullScan", True),  # Publish after full scan
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
    env_cfg.commands.ranges.lin_vel_x = (0.8, 0.8)  # Forward velocity
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)

    # set terrain to USD or default warehouse
    if args_cli.usd_path is not None:
        usd_path = os.path.abspath(args_cli.usd_path)
        print(f"[INFO] Using custom USD environment: {usd_path}")
    else:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
        print(f"[INFO] Using default warehouse USD environment: {usd_path}")

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
        local_position=(0.3, 0.0, 0.35),  # Front of robot, slightly elevated
        local_rotation=(-10.0, 0.0, 0.0),  # Tilted slightly downward
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
            local_position=(0.0, 0.0, 1.0),  # On top of robot pelvis
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
            action = policy(obs)
            obs, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()

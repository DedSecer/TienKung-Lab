# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
OmniGraph setup utilities for ROS2 sensor publishing.

Contains functions for creating OmniGraph action graphs that connect
Isaac Sim sensors to ROS2 topics via the ROS2 bridge extension.
"""

import omni
import omni.kit.app
import omni.graph.core as og
import usdrt.Sdf


def enable_required_extensions():
    """
    Enable all required extensions for ROS2 camera publishing and PhysX LiDAR.

    Returns:
        bool: True if at least one extension was enabled successfully.
    """
    extension_manager = omni.kit.app.get_app().get_extension_manager()

    # All required extensions for ROS2 camera publishing and PhysX LiDAR
    required_extensions = [
        # ROS2 bridge extensions (try new name first, then legacy)
        ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"),
        # Core nodes extensions (try new name first, then legacy)
        ("isaacsim.core.nodes", "omni.isaac.core_nodes"),
        # PhysX sensor extensions for LiDAR
        ("isaacsim.sensors.physx", "omni.isaac.range_sensor"),
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


def setup_ros2_physx_lidar_graph(lidar_prim_path: str, point_cloud_topic: str,
                                 frame_id: str, domain_id: int = 0):
    """
    Setup OmniGraph for publishing PhysX LiDAR data to ROS2 topics.

    Uses IsaacReadLidarPointCloud to read PhysX LiDAR data and
    ROS2PublishPointCloud to publish PointCloud2 messages.

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
    # PhysX LiDAR uses IsaacReadLidarPointCloud + ROS2PublishPointCloud
    node_type_variants = [
        {
            "prefix": "isaacsim",
            "context": "isaacsim.ros2.bridge.ROS2Context",
            "read_lidar_pcl": "isaacsim.sensors.physx.IsaacReadLidarPointCloud",
            "publish_pcl": "isaacsim.ros2.bridge.ROS2PublishPointCloud",
            "read_sim_time": "isaacsim.core.nodes.IsaacReadSimulationTime",
        },
        {
            "prefix": "omni.isaac",
            "context": "omni.isaac.ros2_bridge.ROS2Context",
            "read_lidar_pcl": "omni.isaac.range_sensor.IsaacReadLidarPointCloud",
            "publish_pcl": "omni.isaac.ros2_bridge.ROS2PublishPointCloud",
            "read_sim_time": "omni.isaac.core_nodes.IsaacReadSimulationTime",
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
            print(f"[DEBUG] Trying PhysX lidar node types: {variant['read_lidar_pcl']}")

            # Create the action graph
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2Context", variant["context"]),
                        ("ReadSimTime", variant["read_sim_time"]),
                        ("ReadLidarPCL", variant["read_lidar_pcl"]),
                        ("PublishPCL", variant["publish_pcl"]),
                    ],
                    keys.SET_VALUES: [
                        # ROS2 Context settings
                        ("ROS2Context.inputs:domain_id", domain_id),
                        ("ROS2Context.inputs:useDomainIDEnvVar", False),

                        # PhysX LiDAR prim reference
                        ("ReadLidarPCL.inputs:lidarPrim", [usdrt.Sdf.Path(lidar_prim_path)]),

                        # Point cloud publisher settings
                        ("PublishPCL.inputs:topicName", point_cloud_topic),
                        ("PublishPCL.inputs:frameId", frame_id),
                    ],
                    keys.CONNECT: [
                        # Connect tick to read lidar
                        ("OnPlaybackTick.outputs:tick", "ReadLidarPCL.inputs:execIn"),

                        # Connect lidar output to publisher
                        ("ReadLidarPCL.outputs:execOut", "PublishPCL.inputs:execIn"),
                        ("ReadLidarPCL.outputs:data", "PublishPCL.inputs:data"),

                        # Connect simulation time to publisher
                        ("ReadSimTime.outputs:simulationTime", "PublishPCL.inputs:timeStamp"),

                        # Connect ROS2 context
                        ("ROS2Context.outputs:context", "PublishPCL.inputs:context"),
                    ],
                },
            )

            print(f"[INFO] Created ROS2 PhysX LiDAR graph at: {graph_path}")
            print(f"[INFO] Point cloud topic: {point_cloud_topic}")

            return graph

        except Exception as e:
            last_error = e
            print(f"[DEBUG] Failed with PhysX lidar node types {variant['read_lidar_pcl']}: {e}")
            # Try to clean up the partially created graph
            try:
                og.Controller.delete_graph(graph_path)
            except:
                pass
            continue

    # If all variants failed, raise the last error
    raise RuntimeError(f"Failed to create ROS2 PhysX LiDAR graph with any node type variant. Last error: {last_error}")


def setup_ros2_camera_graph(camera_prim_path: str, rgb_topic: str, depth_topic: str,
                            camera_info_topic: str, frame_id: str,
                            width: int = 640, height: int = 480,
                            domain_id: int = 0):
    """
    Setup OmniGraph for publishing camera data to ROS2 topics.

    Args:
        camera_prim_path: Path to the camera prim
        rgb_topic: ROS2 topic name for RGB image
        depth_topic: ROS2 topic name for depth image
        camera_info_topic: ROS2 topic name for camera info
        frame_id: Frame ID for the camera
        width: Camera image width
        height: Camera image height
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
                        ("CreateRenderProduct.inputs:width", width),
                        ("CreateRenderProduct.inputs:height", height),

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

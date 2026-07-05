# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
Sensor creation utilities for USD environments.

Contains functions for creating cameras and PhysX LiDAR sensors
on robot prims in the USD stage.
"""

import omni
import omni.kit.commands
from pxr import UsdGeom, Gf


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


def create_physx_lidar_on_robot(stage, robot_prim_path: str, lidar_name: str = "mid360_lidar",
                               local_position: tuple = (0.0, 0.0, 0.4),
                               local_rotation: tuple = (0.0, 0.0, 0.0),
                               fov: tuple = (360.0, 30.0),
                               resolution: tuple = (0.4, 4.0),
                               rotation_rate: float = 20.0,
                               valid_range: tuple = (0.4, 100.0),
                               high_lod: bool = True):
    """
    Create a PhysX LiDAR sensor attached to the robot using Isaac Sim's RangeSensor.

    This creates a PhysX-based rotating LiDAR suitable for SLAM and navigation applications.
    PhysX LiDAR uses physics engine raycasting (lighter than RTX LiDAR) and allows
    direct parameter control (FOV, resolution, rotation rate, valid range).

    Args:
        stage: USD stage
        robot_prim_path: Path to the robot prim
        lidar_name: Name for the lidar sensor
        local_position: Local position offset from parent (x, y, z) in meters
        local_rotation: Local rotation offset from parent (roll, pitch, yaw) in degrees
        fov: Field of view (horizontal, vertical) in degrees
        resolution: Angular resolution (horizontal, vertical) in degrees
        rotation_rate: Rotation rate in Hz (0 for static scan)
        valid_range: Valid detection range (min, max) in meters
        high_lod: Enable high LOD for 3D point cloud output

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

    try:
        # Create PhysX LiDAR using RangeSensorCreateLidar command
        print(f"[INFO] Creating PhysX LiDAR: FOV=({fov[0]}, {fov[1]}), Resolution=({resolution[0]}, {resolution[1]}), RotRate={rotation_rate}Hz, Range=({valid_range[0]}, {valid_range[1]})m")

        result, lidar_prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidar_path,
            parent=None,
            min_range=valid_range[0],
            max_range=valid_range[1],
            draw_points=False,
            draw_lines=False,
            horizontal_fov=fov[0],
            vertical_fov=fov[1],
            horizontal_resolution=resolution[0],
            vertical_resolution=resolution[1],
            rotation_rate=rotation_rate,
            high_lod=high_lod,
            yaw_offset=0.0,
            enable_semantics=False,
        )

        if result and lidar_prim:
            # Set the local transform (position offset)
            prim = stage.GetPrimAtPath(lidar_path)
            if prim.IsValid():
                prim.GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(local_position[0], local_position[1], local_position[2])
                )
                # Set rotation if needed
                if any(r != 0 for r in local_rotation):
                    xform = UsdGeom.Xformable(prim)
                    # Check if rotate ops exist, if not create them
                    existing_ops = [op.GetOpType() for op in xform.GetOrderedXformOps()]
                    if UsdGeom.XformOp.TypeRotateXYZ not in existing_ops:
                        rotate_op = xform.AddRotateXYZOp()
                        rotate_op.Set(Gf.Vec3f(local_rotation[0], local_rotation[1], local_rotation[2]))

            print(f"[INFO] Created PhysX LiDAR at: {lidar_path}")
            print(f"[INFO] High LOD (3D point cloud): {high_lod}")
            return lidar_path
        else:
            print(f"[ERROR] Failed to create PhysX LiDAR")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to create PhysX LiDAR: {e}")
        import traceback
        traceback.print_exc()
        return None

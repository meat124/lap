#!/usr/bin/env python3
"""Convert rby1 joint-space dataset to end-effector (EEF) space using FK.

Implements forward kinematics for the rby1a arm chain (base → ee_right)
using parameters extracted from the rby1a URDF, with pure numpy (no rby1_sdk
dependency required).

Usage:
    # Convert all 300-episode dataset
    python scripts/convert_rby1_joint_to_eef.py \
        --input-dir /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2 \
        --output-dir /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2_eef

    # Convert 100-episode subset
    python scripts/convert_rby1_joint_to_eef.py \
        --input-dir /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2_100 \
        --output-dir /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2_100_eef

The script:
  1. Uses the rby1a kinematic chain (base → ee_right) extracted from the URDF
  2. Maps dataset 16D joints → joint values for the chain (torso fixed at 0)
  3. Computes FK: base → ee_right transform
  4. Extracts position [x,y,z] + extrinsic XYZ Euler [roll,pitch,yaw] + gripper
  5. Writes new parquet files with 7D state/action columns
  6. Copies videos and updates metadata

Output action format (7D, matching LAP pre-training convention):
  [x, y, z, roll, pitch, yaw, gripper_right]
  - Position in meters (base frame)
  - Rotation as extrinsic XYZ Euler angles in radians (base frame)
  - Gripper: original value from dataset (0-1)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pure-numpy FK for rby1a (base → ee_right)
# ---------------------------------------------------------------------------
# Kinematic chain extracted from rby1-sdk/models/rby1a/urdf/model.urdf:
#
# base
#  └─ torso_0 (revolute, axis=[1,0,0], xyz=[0,0,0.2805])
#      └─ torso_1 (revolute, axis=[0,1,0], xyz=[0,0,0])
#          └─ torso_2 (revolute, axis=[0,1,0], xyz=[0,0,0.350])
#              └─ torso_3 (revolute, axis=[0,1,0], xyz=[0,0,0.350])
#                  └─ torso_4 (revolute, axis=[1,0,0], xyz=[0,0,0])
#                      └─ torso_5 (revolute, axis=[0,0,1], xyz=[0,0,0.309426548461])
#                          └─ right_arm_0 (revolute, axis=[0,0.93969262,-0.34202014], xyz=[0,-0.220,0.080073451539])
#                              └─ right_arm_1 (revolute, axis=[1,0,0], xyz=[0,0,0])
#                                  └─ right_arm_2 (revolute, axis=[0,0,1], xyz=[0,0,0])
#                                      └─ right_arm_3 (revolute, axis=[0,1,0], xyz=[0.031,0,-0.276])
#                                          └─ right_arm_4 (revolute, axis=[0,0,1], xyz=[-0.031,0,-0.256])
#                                              └─ right_arm_5 (revolute, axis=[0,1,0], xyz=[0,0,0])
#                                                  └─ right_arm_6 (revolute, axis=[0,0,1], xyz=[0,0,0])
#                                                      └─ tool_right (fixed, xyz=[0,0,-0.1087])
#                                                          └─ FT_Sensor_END_right (fixed, xyz=[0,0,-0.0461])  → ee_right
#
# URDF full q-vector (24 DOF):
#   0: right_wheel, 1: left_wheel
#   2-7: torso_0..torso_5
#   8-14: right_arm_0..right_arm_6
#   15-21: left_arm_0..left_arm_6
#   22-23: head_0, head_1
#
# Dataset 16D ordering (from modality.json):
#   0-6: right_arm (7 joints)
#   7-13: left_arm (7 joints)
#   14: right_gripper
#   15: left_gripper

# Chain joint definitions: (xyz_offset, axis, joint_name_or_None_for_fixed)
# Each entry: (translation, rotation_axis, is_fixed)
# Torso joints (indices 0-5) are always 0 during data collection.
_CHAIN = [
    # torso_0: xyz=[0,0,0.2805], axis=[1,0,0]
    (np.array([0.0, 0.0, 0.2805]), np.array([1.0, 0.0, 0.0]), "torso_0"),
    # torso_1: xyz=[0,0,0], axis=[0,1,0]
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "torso_1"),
    # torso_2: xyz=[0,0,0.350], axis=[0,1,0]
    (np.array([0.0, 0.0, 0.350]), np.array([0.0, 1.0, 0.0]), "torso_2"),
    # torso_3: xyz=[0,0,0.350], axis=[0,1,0]
    (np.array([0.0, 0.0, 0.350]), np.array([0.0, 1.0, 0.0]), "torso_3"),
    # torso_4: xyz=[0,0,0], axis=[1,0,0]
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), "torso_4"),
    # torso_5: xyz=[0,0,0.309426548461], axis=[0,0,1]
    (np.array([0.0, 0.0, 0.309426548461]), np.array([0.0, 0.0, 1.0]), "torso_5"),
    # right_arm_0: xyz=[0,-0.220,0.080073451539], axis=[0,0.93969262,-0.34202014]
    (np.array([0.0, -0.220, 0.080073451539]), np.array([0.0, 0.93969262, -0.34202014]), "right_arm_0"),
    # right_arm_1: xyz=[0,0,0], axis=[1,0,0]
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), "right_arm_1"),
    # right_arm_2: xyz=[0,0,0], axis=[0,0,1]
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), "right_arm_2"),
    # right_arm_3: xyz=[0.031,0,-0.276], axis=[0,1,0]
    (np.array([0.031, 0.0, -0.276]), np.array([0.0, 1.0, 0.0]), "right_arm_3"),
    # right_arm_4: xyz=[-0.031,0,-0.256], axis=[0,0,1]
    (np.array([-0.031, 0.0, -0.256]), np.array([0.0, 0.0, 1.0]), "right_arm_4"),
    # right_arm_5: xyz=[0,0,0], axis=[0,1,0]
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "right_arm_5"),
    # right_arm_6: xyz=[0,0,0], axis=[0,0,1]
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), "right_arm_6"),
]

# Fixed transforms after the last revolute joint → ee_right
_TOOL_OFFSET = np.array([0.0, 0.0, -0.1087])       # tool_right
_FT_SENSOR_OFFSET = np.array([0.0, 0.0, -0.0461])   # FT_Sensor_END_right


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation: rotation matrix for angle (rad) around unit axis."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform from 3×3 rotation and 3D translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Actual torso joint angles used during data collection (fixed throughout all episodes).
# Verified from raw H5 files: samples/robot_position[2:8] is constant at these values.
# URDF ordering: torso_0..torso_5
_TORSO_ANGLES_DEG = np.array([0.0, 20.0, -40.0, 35.0, 0.0, 0.0])
_TORSO_ANGLES_RAD = np.deg2rad(_TORSO_ANGLES_DEG)


def forward_kinematics(joint_angles_13: np.ndarray) -> np.ndarray:
    """Compute base → ee_right 4×4 transform.

    Args:
        joint_angles_13: (13,) array of joint angles in radians.
            [torso_0..torso_5 (6), right_arm_0..right_arm_6 (7)]
            Use _TORSO_ANGLES_RAD for the torso values (fixed during data collection).

    Returns:
        T: (4, 4) homogeneous transform from base to ee_right.
    """
    T = np.eye(4)
    for i, (xyz, axis, _name) in enumerate(_CHAIN):
        # Translation
        T_joint = _homogeneous(np.eye(3), xyz)
        # Rotation
        R = _rotation_matrix(axis, joint_angles_13[i])
        T_rot = _homogeneous(R, np.zeros(3))
        T = T @ T_joint @ T_rot

    # Fixed tool + FT sensor offsets
    T = T @ _homogeneous(np.eye(3), _TOOL_OFFSET)
    T = T @ _homogeneous(np.eye(3), _FT_SENSOR_OFFSET)
    return T


def joint_16d_to_eef_7d(joint_16d: np.ndarray) -> np.ndarray:
    """Convert 16D dataset joint vector to 7D EEF pose.

    Dataset ordering: [right_arm(7), left_arm(7), right_gripper(1), left_gripper(1)]

    Returns:
        eef_7d: [x, y, z, roll, pitch, yaw, gripper_right]
    """
    # Build 13D chain input: [torso(6)=actual, right_arm(7)]
    chain_q = np.zeros(13)
    chain_q[0:6] = _TORSO_ANGLES_RAD  # torso fixed at actual data-collection pose
    chain_q[6:13] = joint_16d[0:7]    # right_arm_0..6

    T = forward_kinematics(chain_q)

    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    euler = rot.as_euler("xyz", degrees=False)
    gripper = joint_16d[14]

    return np.concatenate([pos, euler, [gripper]]).astype(np.float32)


def convert_frames_batch(joints_16d: np.ndarray) -> np.ndarray:
    """Convert batch of 16D joint frames to 7D EEF frames.

    Args:
        joints_16d: (N, 16) array.

    Returns:
        eef_7d: (N, 7) array.
    """
    N = joints_16d.shape[0]
    result = np.empty((N, 7), dtype=np.float32)
    for i in range(N):
        result[i] = joint_16d_to_eef_7d(joints_16d[i])
    return result


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def convert_parquet(input_path: Path, output_path: Path) -> int:
    """Convert a single parquet episode file from joint to EEF space."""
    df = pd.read_parquet(input_path)

    if "observation.state" in df.columns:
        state_arr = np.stack(df["observation.state"].values).astype(np.float32)
        df["observation.state"] = list(convert_frames_batch(state_arr))

    if "action" in df.columns:
        action_arr = np.stack(df["action"].values).astype(np.float32)
        df["action"] = list(convert_frames_batch(action_arr))

    if "state" in df.columns:
        state_arr = np.stack(df["state"].values).astype(np.float32)
        df["state"] = list(convert_frames_batch(state_arr))

    if "actions" in df.columns:
        actions_arr = np.stack(df["actions"].values).astype(np.float32)
        df["actions"] = list(convert_frames_batch(actions_arr))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return len(df)


def update_info_json(input_path: Path, output_path: Path) -> None:
    """Update info.json to reflect 7D EEF state/action shapes."""
    with open(input_path) as f:
        info = json.load(f)

    for key in ("state", "actions", "observation.state", "action"):
        if key in info.get("features", {}):
            info["features"][key]["shape"] = [7]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)


def update_modality_json(output_path: Path) -> None:
    """Create modality.json describing 7D EEF dimensions."""
    eef_modality = {
        "state": {
            "ee_pos_x": {"start": 0, "end": 1},
            "ee_pos_y": {"start": 1, "end": 2},
            "ee_pos_z": {"start": 2, "end": 3},
            "ee_rot_roll": {"start": 3, "end": 4},
            "ee_rot_pitch": {"start": 4, "end": 5},
            "ee_rot_yaw": {"start": 5, "end": 6},
            "right_gripper": {"start": 6, "end": 7},
        },
        "action": {
            "ee_pos_x": {"start": 0, "end": 1},
            "ee_pos_y": {"start": 1, "end": 2},
            "ee_pos_z": {"start": 2, "end": 3},
            "ee_rot_roll": {"start": 3, "end": 4},
            "ee_rot_pitch": {"start": 4, "end": 5},
            "ee_rot_yaw": {"start": 5, "end": 6},
            "right_gripper": {"start": 6, "end": 7},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eef_modality, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert rby1 joint-space dataset to EEF space using FK")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to input LeRobot dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output EEF dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Only convert first episode for testing")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("FK chain: base → torso(6, fixed=0) → right_arm(7) → tool → ee_right")

    # Find all parquet files
    parquet_files = sorted(input_dir.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {input_dir / 'data'}")
    print(f"Found {len(parquet_files)} episode parquet files")

    if args.dry_run:
        parquet_files = parquet_files[:1]
        print("DRY RUN: converting only first episode")

    # Convert parquets
    total_frames = 0
    for pf in tqdm(parquet_files, desc="Converting episodes"):
        rel = pf.relative_to(input_dir)
        out_pf = output_dir / rel
        n = convert_parquet(pf, out_pf)
        total_frames += n

    print(f"Converted {total_frames} frames across {len(parquet_files)} episodes")

    # Copy and update metadata
    meta_in = input_dir / "meta"
    meta_out = output_dir / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)

    if (meta_in / "info.json").exists():
        update_info_json(meta_in / "info.json", meta_out / "info.json")
        print("Updated info.json")

    update_modality_json(meta_out / "modality.json")
    print("Updated modality.json")

    for f in meta_in.iterdir():
        if f.name not in ("info.json", "modality.json"):
            dst = meta_out / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                print(f"Copied {f.name}")

    # Symlink videos directory (unchanged)
    videos_in = input_dir / "videos"
    videos_out = output_dir / "videos"
    if videos_in.exists() and not videos_out.exists():
        videos_out.symlink_to(videos_in.resolve())
        print(f"Symlinked videos → {videos_in.resolve()}")

    # Sanity check
    print("\n--- Sanity Check ---")
    first_pq = sorted((output_dir / "data").glob("**/*.parquet"))[0]
    df = pd.read_parquet(first_pq)
    state_col = "observation.state" if "observation.state" in df.columns else "state"
    action_col = "action" if "action" in df.columns else "actions"
    state_0 = np.array(df[state_col].iloc[0])
    action_0 = np.array(df[action_col].iloc[0])
    print(f"First frame state  (7D EEF): {state_0}")
    print(f"  Position [m]:    {state_0[:3]}")
    print(f"  Euler [rad]:     {state_0[3:6]}")
    print(f"  Gripper:         {state_0[6]:.4f}")
    print(f"First frame action (7D EEF): {action_0}")
    print(f"  Position [m]:    {action_0[:3]}")
    print(f"  Euler [rad]:     {action_0[3:6]}")
    print(f"  Gripper:         {action_0[6]:.4f}")

    # Temporal smoothness check
    states = np.stack(df[state_col].values)
    if len(states) > 1:
        deltas = np.diff(states[:, :3], axis=0)
        max_delta = np.abs(deltas).max()
        mean_delta = np.abs(deltas).mean()
        print(f"\nPosition delta stats (consecutive frames):")
        print(f"  Max  absolute delta: {max_delta:.6f} m")
        print(f"  Mean absolute delta: {mean_delta:.6f} m")
        if max_delta > 0.05:
            print("  WARNING: Large position jumps detected! Check FK mapping.")
        else:
            print("  OK: Trajectory appears smooth.")

    print("\nDone!")


if __name__ == "__main__":
    main()

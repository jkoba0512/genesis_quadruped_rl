#!/usr/bin/env python3
"""
Proper Genesis video recording using official Genesis API.
Based on Genesis repository documentation and examples.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Record video using Genesis cameras")
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--output", default="genesis_robot_video.mp4", help="Output video filename"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of simulation steps"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=[1280, 720],
        help="Video resolution (width height)",
    )
    parser.add_argument(
        "--orbit", action="store_true", help="Orbit camera around robot"
    )
    args = parser.parse_args()

    print("=== Genesis Official Video Recording ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Steps: {args.steps}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")

    # Load trained model
    print("\nLoading trained model...")
    model = PPO.load(args.model)
    print("✅ Model loaded!")

    # Initialize Genesis
    print("\nInitializing Genesis...")
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Create scene in headless mode for video recording
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,  # 50 FPS simulation
            substeps=10,
        ),
        show_viewer=False,  # Critical: no GUI for video recording
        renderer=gs.renderers.Rasterizer(),  # Better performance than ray tracer
    )

    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # Add robot (Go2 quadruped)
    project_root = Path(__file__).parent.parent
    urdf_path = project_root / "assets/robots/go2/urdf/go2_genesis.urdf"

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0, 0, 0.6),  # Higher position for Go2
        )
    )

    # Add cameras for different angles
    print("Setting up cameras...")

    # Main camera (front view)
    cam_main = scene.add_camera(
        res=(args.resolution[0], args.resolution[1]),
        pos=(3.5, -2.0, 2.0),
        lookat=(0.0, 0.0, 0.8),
        fov=45,
        GUI=False,  # Critical for video recording
    )

    # Side camera
    cam_side = scene.add_camera(
        res=(args.resolution[0], args.resolution[1]),
        pos=(0.0, -4.0, 1.5),
        lookat=(0.0, 0.0, 0.8),
        fov=45,
        GUI=False,
    )

    print("Building scene...")
    scene.build()

    # Apply robot grounding for Go2
    print("Applying robot grounding...")
    sys.path.insert(0, str(project_root / "src"))
    try:
        from genesis_quadruped_rl.robots.go2_loader import Go2Robot
        # Use Go2 specific grounding
        go2_loader = Go2Robot(scene, position=None, use_grounding=True)
        grounding_height = 0.1  # Go2 natural height
        robot.set_pos([0, 0, grounding_height])
    except Exception as e:
        print(f"Using default grounding: {e}")
        grounding_height = 0.3
        robot.set_pos([0, 0, grounding_height])

    # Stabilize robot
    for _ in range(10):
        scene.step()

    print(f"✅ Robot loaded at height {grounding_height:.3f}m")

    # Start recording
    print(f"\nStarting video recording...")
    cam_main.start_recording()

    # Create environment wrapper for RL model
    print("Creating environment wrapper...")

    # Go2 observation extraction (matching the trained environment)
    def get_observation():
        pos = robot.get_pos().cpu().numpy()
        quat = robot.get_quat().cpu().numpy()
        joint_pos = robot.get_dofs_position().cpu().numpy()
        joint_vel = robot.get_dofs_velocity().cpu().numpy()

        # Get only controllable joints (12 for Go2)
        controllable_indices = list(range(1, 13))  # Skip base joint
        controllable_joint_pos = joint_pos[controllable_indices]
        controllable_joint_vel = joint_vel[controllable_indices]

        # Match the expected observation space (46 dimensions)
        # Structure: [pos(3), quat(4), joint_pos(12), joint_vel(12), prev_action(12), target_vel(1), padding(2)]
        prev_action = np.zeros(12)  # 12 controllable joints
        target_vel = np.array([0.5])  # Target walking speed
        padding = np.zeros(2)  # Add padding to reach 46 dimensions

        observation = np.concatenate([
            pos,                    # 3
            quat,                   # 4
            controllable_joint_pos, # 12
            controllable_joint_vel, # 12
            prev_action,           # 12
            target_vel,            # 1
            padding                # 2
        ])
        
        print(f"Observation shape: {observation.shape}")  # Debug
        return observation

    # Simulation and recording loop
    print(f"Recording {args.steps} steps...")
    total_reward = 0

    for step in range(args.steps):
        # Get observation
        obs = get_observation()

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Apply action to robot (Go2 control)
        current_pos = robot.get_dofs_position().cpu()  # Move to CPU first
        
        # For Go2, apply action only to controllable joints (skip base joint)
        controllable_indices = list(range(1, 13))  # Skip base joint
        target_pos = current_pos.clone()
        
        # Apply actions to controllable joints only
        for i, joint_idx in enumerate(controllable_indices):
            if i < len(action):
                target_pos[joint_idx] += action[i] * 0.05  # Smaller position changes for stability
        
        robot.control_dofs_position(target_pos)

        # Step simulation
        scene.step()

        # Optional: orbit camera
        if args.orbit:
            angle = step * 2 * np.pi / args.steps
            radius = 4.0
            cam_x = radius * np.cos(angle)
            cam_y = radius * np.sin(angle)
            cam_main.set_pose(pos=(cam_x, cam_y, 2.5), lookat=(0.0, 0.0, 0.8))

        # Render frame (this captures it for the video)
        cam_main.render()

        # Progress update
        if step % 50 == 0:
            pos = robot.get_pos().cpu().numpy()
            print(f"  Step {step}: robot at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Stop recording and save
    print(f"\nSaving video to {args.output}...")
    try:
        cam_main.stop_recording(save_to_filename=args.output, fps=args.fps)
        print("✅ Video saved successfully!")

        # Check file
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print(f"   File: {args.output}")
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Duration: {args.steps/args.fps:.1f} seconds")
        else:
            print("⚠️  Video file not found - Genesis may have saved elsewhere")

    except Exception as e:
        print(f"❌ Error saving video: {e}")
        print("This can happen due to:")
        print("  - Missing video encoding libraries")
        print("  - GPU driver issues")
        print("  - File permission problems")

    print(f"\n🎬 To play the video:")
    print(f"   vlc {args.output}")
    print(f"   mpv {args.output}")
    print(f"   python -m http.server 8000  # Then open browser to localhost:8000")


if __name__ == "__main__":
    main()

"""
Quadruped walking environment using Genesis physics engine.
Adapted from humanoid environment for Unitree Go2 quadruped robot.
"""

import numpy as np
import genesis as gs
from gymnasium import Env, spaces
from typing import Any, Dict, Optional, Tuple
import os
import sys
import torch

# Add project paths
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from genesis_quadruped_rl.robots.go2_loader import Go2Robot
from src.genesis_humanoid_rl.physics.robot_grounding import RobotGroundingFactory


class QuadrupedWalkingEnv(Env):
    """
    Quadruped walking environment for reinforcement learning.

    Uses Genesis physics engine with Unitree Go2 quadruped robot.
    Integrates robot_grounding library for automatic positioning.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        simulation_fps: int = 100,
        control_freq: int = 20,
        episode_length: int = 1000,
        target_velocity: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.control_freq = control_freq
        self.episode_length = episode_length
        self.target_velocity = target_velocity

        # Initialize Genesis only if not already initialized
        try:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        except Exception as e:
            if "Genesis already initialized" not in str(e):
                raise

        # Scene and robot will be initialized in reset()
        self.scene = None
        self.go2_robot = None  # Go2Robot wrapper
        self.robot = None      # Genesis robot entity
        self.current_step = 0

        # Go2-specific constants
        self.num_joints = 12  # Go2 has 12 controllable DOF
        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
        ]

        # Action space: 12 DOF for Go2 (4 legs × 3 joints)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Observation space: 
        # Position (3) + orientation (4) + joint positions (12) + joint velocities (12)
        # + previous action (12) + target velocity (1) = 44 dimensions
        obs_dim = 3 + 4 + 12 + 12 + 12 + 1  # = 44
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.previous_action = np.zeros(self.num_joints)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Clean up existing scene to prevent memory leaks
        if self.scene is not None:
            try:
                # Clear GPU memory from previous scene
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                # Note: Genesis automatically handles scene cleanup
            except:
                pass

        # Create new scene
        scene_kwargs = {
            "sim_options": gs.options.SimOptions(
                dt=1.0 / self.simulation_fps,
                substeps=2,  # Reduce substeps to save memory
            ),
            "show_viewer": False,  # Disable viewer for faster initialization
        }

        if self.render_mode is not None:
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                res=(1024, 768),
                max_FPS=self.simulation_fps,
            )

        self.scene = gs.Scene(**scene_kwargs)

        # Load ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Load Unitree Go2 robot using stable initialization
        # FIXED: Disable grounding to prevent post-grounding instability
        self.go2_robot = Go2Robot(
            self.scene, 
            position=np.array([0.0, 0.0, 0.6]),  # Start higher to prevent ground collision
            pose="standing",  # Natural standing pose
            use_grounding=False  # Disable grounding - it causes post-settling instability
        )
        self.robot = self.go2_robot.robot  # Genesis robot entity

        # Build scene
        self.scene.build()

        # CRITICAL: Apply natural pose to ensure proper joint configuration
        print("Initializing Go2 robot in RL environment...")
        self.go2_robot.apply_natural_pose_and_grounding()
        
        # Check final height
        final_state = self.go2_robot.get_state()
        print(f"✅ Go2 ready for RL at height: {final_state['base_pos'][2]:.3f}m")

        print(f"Loaded Go2 robot with {self.num_joints} controllable DOFs")
        print(f"Robot has {self.robot.n_links} links")

        self.current_step = 0
        self.previous_action = np.zeros(self.num_joints)

        # Initialize previous position for velocity calculation
        if self.robot is not None:
            self._previous_pos = self.robot.get_pos().cpu().numpy()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply action to robot joints using Go2Robot wrapper
        if self.go2_robot is not None:
            # Use the Go2Robot's RL action control method
            self.go2_robot.set_joint_targets_from_actions(action)

        # Step simulation
        steps_per_control = self.simulation_fps // self.control_freq
        for _ in range(steps_per_control):
            self.scene.step()

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.episode_length

        # Store previous action
        self.previous_action = action.copy()

        info = {
            "step": self.current_step,
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from the environment."""
        if self.robot is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get robot base position and orientation
        pos = self.robot.get_pos().cpu().numpy()  # (3,)
        quat = self.robot.get_quat().cpu().numpy()  # (4,) quaternion (x, y, z, w)

        # Get joint positions and velocities (12 controllable joints)
        joint_pos = self.go2_robot.get_joint_positions()  # (12,)
        joint_vel = self.go2_robot.get_joint_velocities()  # (12,)

        # Concatenate all observations
        obs = np.concatenate([
            pos,                     # Base position (3)
            quat,                    # Base orientation (4)
            joint_pos,               # Joint positions (12)
            joint_vel,               # Joint velocities (12)
            self.previous_action,    # Previous action (12)
            [self.target_velocity],  # Target velocity (1)
        ])

        return obs.astype(np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for the current state and action - adapted for quadruped."""
        if self.robot is None:
            return 0.0

        total_reward = 0.0

        # Get current robot state
        pos = self.robot.get_pos().cpu().numpy()
        quat = self.robot.get_quat().cpu().numpy()
        joint_vel = self.go2_robot.get_joint_velocities()

        # 1. Forward velocity reward (primary objective)
        # Encourage movement in positive X direction
        if hasattr(self, "_previous_pos"):
            dt = 1.0 / self.control_freq
            forward_velocity = (pos[0] - self._previous_pos[0]) / dt
            velocity_reward = min(
                forward_velocity / self.target_velocity, 2.0
            )  # Cap at 2x target
            total_reward += 1.0 * velocity_reward

        # 2. Stability reward (keep robot level)
        # For quadrupeds, focus on preventing rolling/pitching
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            # Check roll and pitch (less critical for quadrupeds than humanoids)
            tilt_magnitude = np.sqrt(quat[0] ** 2 + quat[1] ** 2)
            stability_reward = max(0.0, 1.0 - 3.0 * tilt_magnitude)  # More lenient than humanoid
        else:
            stability_reward = 0.0
        total_reward += 0.3 * stability_reward  # Reduced weight vs humanoid

        # 3. Height maintenance reward (quadruped-specific)
        # Go2 should maintain ~0.36m height (natural settled height)
        target_height = 0.36  # CORRECTED: Go2 natural standing height
        height_diff = abs(pos[2] - target_height)
        height_reward = max(0.0, 1.0 - 3.0 * height_diff)  # More sensitive to height changes
        total_reward += 0.4 * height_reward  # Higher weight for quadruped

        # 4. Energy efficiency penalty
        # Penalize excessive joint velocities
        joint_vel_penalty = np.mean(np.square(joint_vel))
        energy_penalty = min(joint_vel_penalty, 10.0)  # Cap penalty
        total_reward -= 0.1 * energy_penalty

        # 5. Action smoothness reward
        # Encourage smooth gait patterns
        if self.previous_action is not None:
            action_diff = np.mean(np.square(action - self.previous_action))
            smoothness_penalty = min(action_diff, 2.0)
            total_reward -= 0.1 * smoothness_penalty

        # 6. Quadruped-specific: Gait symmetry reward
        # Encourage symmetric movement of left/right legs
        if len(action) >= 12:
            # Compare left vs right legs (FL,FR vs RL,RR)
            left_legs = action[[0,1,2, 6,7,8]]    # FL + RL
            right_legs = action[[3,4,5, 9,10,11]] # FR + RR
            symmetry_diff = np.mean(np.abs(left_legs - right_legs))
            symmetry_reward = max(0.0, 1.0 - 2.0 * symmetry_diff)
            total_reward += 0.2 * symmetry_reward

        # 7. Ground contact penalty (quadruped should stay grounded)
        # Penalize if robot jumps too high (more lenient during initial settling)
        height_penalty_threshold = 2.0 if self.current_step < 10 else 0.6
        if pos[2] > height_penalty_threshold:
            total_reward -= 0.5

        # Store current position for next velocity calculation
        self._previous_pos = pos.copy()

        return total_reward

    def _is_terminated(self) -> bool:
        """Check if episode should be terminated - adapted for quadruped."""
        if self.robot is None:
            return True

        # Get current robot state
        pos = self.robot.get_pos().cpu().numpy()
        quat = self.robot.get_quat().cpu().numpy()

        # 1. Height termination - quadruped fell down or flipped
        if pos[2] < 0.05:  # Allow robot to settle lower while still upright
            return True

        # 2. Extreme tilt termination - quadruped flipped over
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            # Convert quaternion to euler angles for proper tilt detection
            # For quaternion [w, x, y, z], extract pitch and roll
            # Roll (x-axis rotation): atan2(2(wy + xz), 1 - 2(y² + x²))
            # Pitch (y-axis rotation): asin(2(wx - yz))
            w, x, y, z = quat
            
            # Calculate roll and pitch angles
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
            
            # Check if robot is flipped (more than 60 degrees tilt in any direction)
            max_tilt_radians = np.pi / 3  # 60 degrees
            if abs(roll) > max_tilt_radians or abs(pitch) > max_tilt_radians:
                return True

        # 3. Out of bounds termination
        if abs(pos[0]) > 10.0 or abs(pos[1]) > 5.0:
            return True

        # 4. Excessive height termination (unrealistic jumping)
        # More lenient during initial steps to allow grounding to settle
        height_threshold = 3.0 if self.current_step < 10 else 1.0
        if pos[2] > height_threshold:
            return True

        return False

    def render(self):
        """Render the environment (Genesis handles this automatically)."""
        if self.render_mode is not None and self.scene is not None:
            # Genesis automatically renders when viewer is enabled
            pass

    def close(self):
        """Clean up the environment."""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Reset scene reference
            self.scene = None
            self.robot = None
            self.go2_robot = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            # Don't fail on cleanup errors
            pass
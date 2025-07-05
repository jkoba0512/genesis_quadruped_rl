#!/usr/bin/env python3
"""
Enhanced quadruped environment with advanced reward functions.
Combines the validated RL pipeline with sophisticated quadruped rewards.
"""

import numpy as np
from gymnasium import Env, spaces
from typing import Any, Dict, Optional, Tuple

from ..rewards.quadruped_rewards import QuadrupedRewardCalculator, RewardWeights


class EnhancedQuadrupedEnv(Env):
    """
    Enhanced quadruped environment with sophisticated reward system.
    Uses simplified physics for stability with advanced reward functions.
    """
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 simulation_fps: int = 100,
                 control_freq: int = 20,
                 episode_length: int = 1000,
                 target_velocity: float = 1.0,
                 target_height: float = 0.3,
                 reward_weights: Optional[RewardWeights] = None,
                 **kwargs):
        super().__init__()
        
        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.control_freq = control_freq
        self.episode_length = episode_length
        self.target_velocity = target_velocity
        self.target_height = target_height
        
        # Enhanced reward system
        self.reward_calculator = QuadrupedRewardCalculator(
            target_velocity=target_velocity,
            target_height=target_height,
            weights=reward_weights
        )
        
        # Environment state
        self.current_step = 0
        self.max_steps = episode_length
        
        # Robot state (simplified physics)
        self.base_pos = np.array([0.0, 0.0, target_height])
        self.base_vel = np.array([0.0, 0.0, 0.0])
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        
        # Environment dynamics parameters
        self.dt = 1.0 / simulation_fps
        self.mass = 15.0  # kg (approximate Go2 mass)
        self.inertia = np.eye(3) * 0.5  # Simplified inertia
        
        # Action and observation spaces
        # Actions: 12 joint position targets (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Observations: base_pos(3) + base_vel(3) + base_quat(4) + joint_pos(12) + joint_vel(12) + prev_action(12) = 46
        obs_dim = 3 + 3 + 4 + 12 + 12 + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.previous_action = np.zeros(12)
        
        # Training metrics
        self.episode_reward = 0.0
        self.episode_length_actual = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset robot state
        self.base_pos = np.array([0.0, 0.0, self.target_height])
        self.base_vel = np.array([0.0, 0.0, 0.0])
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Initialize joints to stable configuration
        self.joint_pos = np.array([0.0, 0.5, -1.0] * 4)  # Stable standing pose
        self.joint_vel = np.zeros(12)
        
        self.current_step = 0
        self.previous_action = np.zeros(12)
        self.episode_reward = 0.0
        self.episode_length_actual = 0
        
        # Reset reward calculator state
        self.reward_calculator.prev_position = None
        self.reward_calculator.prev_action = None
        self.reward_calculator.prev_joint_positions = None
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        self.episode_length_actual += 1
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action to robot (simplified dynamics)
        self._apply_action(action)
        
        # Update physics simulation
        self._update_physics()
        
        # Calculate reward using advanced reward system
        robot_state = self._get_robot_state()
        reward, reward_components = self.reward_calculator.calculate_reward(
            robot_state=robot_state,
            action=action,
            dt=self.dt,
            step_count=self.current_step
        )
        
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Store previous action
        self.previous_action = action.copy()
        
        # Create info dictionary with reward components
        info = {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "reward_components": reward_components,
            "robot_state": robot_state,
            "terminated": terminated,
            "truncated": truncated,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to robot joints (simplified control)."""
        # Convert normalized action [-1, 1] to joint targets
        # Each joint has specific range based on Go2 specifications
        joint_ranges = np.array([
            # Hip joints: ±60°
            [-1.0472, 1.0472],   # FL_hip
            [-1.0472, 1.0472],   # FR_hip  
            [-1.0472, 1.0472],   # RL_hip
            [-1.0472, 1.0472],   # RR_hip
            # Thigh joints: -90° to 200°
            [-1.5708, 3.4907],   # FL_thigh
            [-1.5708, 3.4907],   # FR_thigh
            [-1.5708, 3.4907],   # RL_thigh
            [-1.5708, 3.4907],   # RR_thigh
            # Calf joints: -156° to -48°
            [-2.7227, -0.83776], # FL_calf
            [-2.7227, -0.83776], # FR_calf
            [-2.7227, -0.83776], # RL_calf
            [-2.7227, -0.83776], # RR_calf
        ])
        
        # Map actions to joint targets
        joint_targets = np.zeros(12)
        for i in range(12):
            low, high = joint_ranges[i]
            joint_targets[i] = low + (action[i] + 1.0) / 2.0 * (high - low)
        
        # Simple PD control to reach targets
        kp = 10.0  # Position gain
        kd = 1.0   # Velocity gain
        
        joint_error = joint_targets - self.joint_pos
        joint_forces = kp * joint_error - kd * self.joint_vel
        
        # Update joint velocities and positions (simplified integration)
        joint_acceleration = joint_forces / 0.1  # Simplified joint inertia
        self.joint_vel += joint_acceleration * self.dt
        self.joint_pos += self.joint_vel * self.dt
        
        # Apply joint limits
        for i in range(12):
            low, high = joint_ranges[i]
            self.joint_pos[i] = np.clip(self.joint_pos[i], low, high)
    
    def _update_physics(self):
        """Update robot physics (simplified model)."""
        # Estimate base motion from leg movements
        # This is a simplified model - real physics would be much more complex
        
        # Calculate center of mass motion from leg movements
        leg_motion = np.mean(np.abs(self.joint_vel[:6]))  # Front legs primarily
        
        # Forward motion based on leg coordination
        if leg_motion > 0.1:
            # Simple walking model: leg motion creates forward thrust
            thrust = min(leg_motion * 2.0, 3.0)  # Cap maximum thrust
            self.base_vel[0] += thrust * self.dt / self.mass
        
        # Apply drag
        drag_coeff = 0.5
        self.base_vel *= (1.0 - drag_coeff * self.dt)
        
        # Update position
        self.base_pos += self.base_vel * self.dt
        
        # Simple height dynamics based on leg extension
        avg_leg_extension = 0.0
        for i in range(4):
            thigh_angle = self.joint_pos[i + 4]  # Thigh joints start at index 4
            calf_angle = self.joint_pos[i + 8]   # Calf joints start at index 8
            
            # Simplified leg length calculation
            leg_length = 0.2 + 0.15 * np.cos(thigh_angle) + 0.15 * np.cos(thigh_angle + calf_angle)
            avg_leg_extension += leg_length
        
        avg_leg_extension /= 4.0
        
        # Target height based on leg extension
        target_z = avg_leg_extension
        
        # Simple height control
        height_error = target_z - self.base_pos[2]
        self.base_vel[2] += height_error * 5.0 * self.dt  # Height control gain
        
        # Apply gravity
        self.base_vel[2] -= 9.81 * self.dt
        
        # Ground contact (prevent going below ground)
        if self.base_pos[2] < 0.05:
            self.base_pos[2] = 0.05
            self.base_vel[2] = max(0.0, self.base_vel[2])
        
        # Simple orientation dynamics (keep level)
        # In real implementation, this would be much more complex
        orientation_damping = 0.95
        self.base_quat[0] *= orientation_damping  # Roll
        self.base_quat[1] *= orientation_damping  # Pitch
        # Keep yaw (heading) stable
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(self.base_quat)
        if quat_norm > 0:
            self.base_quat /= quat_norm
    
    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state for reward calculation."""
        return {
            'base_pos': self.base_pos.copy(),
            'base_vel': self.base_vel.copy(),
            'base_quat': self.base_quat.copy(),
            'joint_pos': self.joint_pos.copy(),
            'joint_vel': self.joint_vel.copy(),
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.base_pos,           # 3
            self.base_vel,           # 3
            self.base_quat,          # 4
            self.joint_pos,          # 12
            self.joint_vel,          # 12
            self.previous_action,    # 12
        ])
        return obs.astype(np.float32)
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Height termination
        if self.base_pos[2] < 0.1 or self.base_pos[2] > 1.0:
            return True
        
        # Extreme tilt termination
        quat_norm = np.linalg.norm(self.base_quat)
        if quat_norm > 0:
            tilt_magnitude = np.sqrt(self.base_quat[0]**2 + self.base_quat[1]**2)
            if tilt_magnitude > 0.8:  # About 50 degrees
                return True
        
        # Out of bounds
        if abs(self.base_pos[0]) > 50.0 or abs(self.base_pos[1]) > 10.0:
            return True
        
        return False
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            # In a real implementation, this would show the robot
            pass
    
    def close(self):
        """Clean up the environment."""
        pass
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training metrics for monitoring."""
        return {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length_actual,
            "average_reward_per_step": self.episode_reward / max(1, self.episode_length_actual),
            "forward_distance": self.base_pos[0],
            "final_height": self.base_pos[2],
            "final_velocity": self.base_vel[0],
        }
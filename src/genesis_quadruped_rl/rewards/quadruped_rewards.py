#!/usr/bin/env python3
"""
Advanced reward functions for quadruped locomotion.
Designed specifically for Go2 quadruped robot walking patterns.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Configuration for reward component weights."""
    forward_velocity: float = 1.0
    stability: float = 0.4
    height_maintenance: float = 0.6
    energy_efficiency: float = 0.1
    action_smoothness: float = 0.1
    gait_symmetry: float = 0.3
    foot_contact: float = 0.2
    orientation: float = 0.2
    contact_force: float = 0.15
    leg_coordination: float = 0.25


class QuadrupedRewardCalculator:
    """
    Advanced reward calculator for quadruped locomotion.
    Encourages natural walking gaits and stable movement patterns.
    """
    
    def __init__(self, 
                 target_velocity: float = 1.0,
                 target_height: float = 0.3,
                 weights: Optional[RewardWeights] = None):
        """
        Initialize reward calculator.
        
        Args:
            target_velocity: Target forward velocity (m/s)
            target_height: Target body height (m)
            weights: Reward component weights
        """
        self.target_velocity = target_velocity
        self.target_height = target_height
        self.weights = weights or RewardWeights()
        
        # Gait timing for trotting pattern
        self.trot_phase_offset = np.pi  # Diagonal legs out of phase
        
        # Previous state for derivatives
        self.prev_position = None
        self.prev_action = None
        self.prev_joint_positions = None
        
        # Contact history for gait analysis
        self.contact_history = []
        self.max_contact_history = 20
        
    def calculate_reward(self, 
                        robot_state: Dict[str, np.ndarray],
                        action: np.ndarray,
                        dt: float = 0.02,
                        step_count: int = 0) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward for quadruped locomotion.
        
        Args:
            robot_state: Current robot state (position, velocity, joints, etc.)
            action: Current action taken
            dt: Time step
            step_count: Current step in episode
            
        Returns:
            Tuple of (total_reward, component_rewards)
        """
        components = {}
        
        # Extract state information
        base_pos = robot_state.get('base_pos', np.zeros(3))
        base_vel = robot_state.get('base_vel', np.zeros(3))
        base_quat = robot_state.get('base_quat', np.array([0, 0, 0, 1]))
        joint_pos = robot_state.get('joint_pos', np.zeros(12))
        joint_vel = robot_state.get('joint_vel', np.zeros(12))
        
        # 1. Forward Velocity Reward
        components['forward_velocity'] = self._forward_velocity_reward(base_vel)
        
        # 2. Stability Reward (orientation and height)
        components['stability'] = self._stability_reward(base_quat)
        
        # 3. Height Maintenance Reward
        components['height_maintenance'] = self._height_maintenance_reward(base_pos)
        
        # 4. Energy Efficiency Reward
        components['energy_efficiency'] = self._energy_efficiency_reward(joint_vel, action)
        
        # 5. Action Smoothness Reward
        components['action_smoothness'] = self._action_smoothness_reward(action)
        
        # 6. Gait Symmetry Reward
        components['gait_symmetry'] = self._gait_symmetry_reward(action, step_count)
        
        # 7. Foot Contact Reward (estimated)
        components['foot_contact'] = self._foot_contact_reward(joint_pos, base_pos)
        
        # 8. Base Orientation Reward
        components['orientation'] = self._orientation_reward(base_quat)
        
        # 9. Contact Force Reward (estimated)
        components['contact_force'] = self._contact_force_reward(joint_pos, base_pos)
        
        # 10. Leg Coordination Reward
        components['leg_coordination'] = self._leg_coordination_reward(action, joint_pos)
        
        # Calculate weighted total
        total_reward = 0.0
        for component, value in components.items():
            weight = getattr(self.weights, component, 0.0)
            total_reward += weight * value
        
        # Update previous state
        self._update_previous_state(base_pos, action, joint_pos)
        
        return total_reward, components
    
    def _forward_velocity_reward(self, base_vel: np.ndarray) -> float:
        """Reward forward movement towards target velocity."""
        forward_vel = base_vel[0]
        
        # Optimal velocity reward
        vel_error = abs(forward_vel - self.target_velocity)
        optimal_reward = max(0.0, 1.0 - vel_error / self.target_velocity)
        
        # Bonus for being in target range
        if 0.8 * self.target_velocity <= forward_vel <= 1.2 * self.target_velocity:
            optimal_reward += 0.5
        
        # Penalty for backwards movement
        if forward_vel < 0:
            optimal_reward -= 2.0 * abs(forward_vel)
        
        return np.clip(optimal_reward, -2.0, 2.0)
    
    def _stability_reward(self, base_quat: np.ndarray) -> float:
        """Reward stable upright orientation."""
        if len(base_quat) != 4:
            return 0.0
        
        # Calculate roll and pitch from quaternion
        # quat = [x, y, z, w]
        roll = np.arcsin(2 * (base_quat[3] * base_quat[0] + base_quat[1] * base_quat[2]))
        pitch = np.arctan2(2 * (base_quat[3] * base_quat[1] - base_quat[2] * base_quat[0]),
                          1 - 2 * (base_quat[0]**2 + base_quat[1]**2))
        
        # Reward small angles
        roll_reward = max(0.0, 1.0 - abs(roll) / (np.pi / 6))  # 30 degree tolerance
        pitch_reward = max(0.0, 1.0 - abs(pitch) / (np.pi / 6))  # 30 degree tolerance
        
        return (roll_reward + pitch_reward) / 2.0
    
    def _height_maintenance_reward(self, base_pos: np.ndarray) -> float:
        """Reward maintaining target height."""
        current_height = base_pos[2]
        height_error = abs(current_height - self.target_height)
        
        # More sensitive to height changes for quadrupeds
        height_reward = max(0.0, 1.0 - height_error / (self.target_height * 0.5))
        
        # Bonus for being very close to target
        if height_error < 0.05:  # Within 5cm
            height_reward += 0.5
        
        return height_reward
    
    def _energy_efficiency_reward(self, joint_vel: np.ndarray, action: np.ndarray) -> float:
        """Penalize excessive energy usage."""
        # Joint velocity penalty
        vel_penalty = np.mean(np.square(joint_vel))
        
        # Action magnitude penalty
        action_penalty = np.mean(np.square(action))
        
        # Combined energy penalty
        energy_penalty = 0.5 * vel_penalty + 0.5 * action_penalty
        
        return -min(energy_penalty, 5.0)  # Cap penalty
    
    def _action_smoothness_reward(self, action: np.ndarray) -> float:
        """Reward smooth action transitions."""
        if self.prev_action is None:
            return 0.0
        
        action_diff = np.mean(np.square(action - self.prev_action))
        smoothness_reward = max(0.0, 1.0 - action_diff / 2.0)
        
        return smoothness_reward
    
    def _gait_symmetry_reward(self, action: np.ndarray, step_count: int) -> float:
        """Reward symmetric gait patterns."""
        if len(action) < 12:
            return 0.0
        
        # Split into leg groups (each leg has 3 joints: hip, thigh, calf)
        fl_leg = action[0:3]   # Front Left
        fr_leg = action[3:6]   # Front Right  
        rl_leg = action[6:9]   # Rear Left
        rr_leg = action[9:12]  # Rear Right
        
        # Trotting gait: diagonal legs should be coordinated
        # FL + RR vs FR + RL
        diagonal1 = np.concatenate([fl_leg, rr_leg])  # FL + RR
        diagonal2 = np.concatenate([fr_leg, rl_leg])  # FR + RL
        
        # Calculate phase difference for trotting
        phase = (step_count * 0.1) % (2 * np.pi)
        
        # Expected trot pattern: diagonal pairs 180° out of phase
        expected_diff = np.sin(phase) - np.sin(phase + np.pi)
        actual_diff = np.mean(diagonal1) - np.mean(diagonal2)
        
        # Reward when actual matches expected pattern
        phase_reward = max(0.0, 1.0 - abs(actual_diff - expected_diff))
        
        # Left-right symmetry within front/rear pairs
        front_symmetry = 1.0 - np.mean(np.abs(fl_leg - fr_leg))
        rear_symmetry = 1.0 - np.mean(np.abs(rl_leg - rr_leg))
        symmetry_reward = (front_symmetry + rear_symmetry) / 2.0
        
        return (phase_reward + symmetry_reward) / 2.0
    
    def _foot_contact_reward(self, joint_pos: np.ndarray, base_pos: np.ndarray) -> float:
        """Estimate and reward proper foot contact patterns."""
        if len(joint_pos) < 12:
            return 0.0
        
        # Estimate foot heights based on joint positions
        # This is a simplified calculation - in real implementation would use forward kinematics
        foot_heights = []
        
        for leg_idx in range(4):
            hip_angle = joint_pos[leg_idx * 3 + 0]      # Hip
            thigh_angle = joint_pos[leg_idx * 3 + 1]    # Thigh
            calf_angle = joint_pos[leg_idx * 3 + 2]     # Calf
            
            # Simplified leg length calculation
            leg_extension = 0.4 * (np.cos(thigh_angle) + np.cos(thigh_angle + calf_angle))
            foot_height = base_pos[2] - leg_extension
            foot_heights.append(foot_height)
        
        foot_heights = np.array(foot_heights)
        
        # Reward feet being close to ground (contact)
        contact_reward = np.mean(np.exp(-10 * np.maximum(foot_heights, 0.0)))
        
        # Penalty for feet going underground
        penetration_penalty = np.sum(np.maximum(-foot_heights, 0.0)) * 10
        
        return contact_reward - penetration_penalty
    
    def _orientation_reward(self, base_quat: np.ndarray) -> float:
        """Reward maintaining proper heading direction."""
        if len(base_quat) != 4:
            return 0.0
        
        # Calculate yaw (heading direction)
        yaw = np.arctan2(2 * (base_quat[3] * base_quat[2] + base_quat[0] * base_quat[1]),
                        1 - 2 * (base_quat[1]**2 + base_quat[2]**2))
        
        # Reward maintaining forward heading (yaw ≈ 0)
        heading_reward = max(0.0, 1.0 - abs(yaw) / (np.pi / 4))  # 45 degree tolerance
        
        return heading_reward
    
    def _contact_force_reward(self, joint_pos: np.ndarray, base_pos: np.ndarray) -> float:
        """Estimate and reward appropriate ground contact forces."""
        # This is a simplified estimation - would need actual force sensors in real implementation
        
        # Estimate leg loading based on joint positions
        leg_loads = []
        
        for leg_idx in range(4):
            if len(joint_pos) >= (leg_idx + 1) * 3:
                thigh_angle = joint_pos[leg_idx * 3 + 1]
                calf_angle = joint_pos[leg_idx * 3 + 2]
                
                # Estimate leg compression (more compression = more load)
                compression = max(0.0, 0.5 - (thigh_angle + abs(calf_angle)) / 4.0)
                leg_loads.append(compression)
        
        if not leg_loads:
            return 0.0
        
        leg_loads = np.array(leg_loads)
        
        # Reward balanced loading across legs
        load_balance = 1.0 - np.std(leg_loads)
        
        # Reward moderate total loading (not too stiff, not too loose)
        total_load = np.mean(leg_loads)
        load_magnitude = max(0.0, 1.0 - abs(total_load - 0.3) / 0.3)
        
        return (load_balance + load_magnitude) / 2.0
    
    def _leg_coordination_reward(self, action: np.ndarray, joint_pos: np.ndarray) -> float:
        """Reward coordinated leg movements."""
        if len(action) < 12:
            return 0.0
        
        # Calculate coordination between joints within each leg
        leg_coordination = 0.0
        
        for leg_idx in range(4):
            hip_action = action[leg_idx * 3 + 0]
            thigh_action = action[leg_idx * 3 + 1]
            calf_action = action[leg_idx * 3 + 2]
            
            # Hip and thigh should be somewhat coordinated for natural walking
            hip_thigh_coord = 1.0 - abs(hip_action - thigh_action * 0.5) / 2.0
            
            # Thigh and calf should be coordinated (knee joint action)
            thigh_calf_coord = 1.0 - abs(thigh_action + calf_action) / 2.0
            
            leg_coordination += (hip_thigh_coord + thigh_calf_coord) / 2.0
        
        return leg_coordination / 4.0  # Average across all legs
    
    def _update_previous_state(self, position: np.ndarray, action: np.ndarray, joint_pos: np.ndarray):
        """Update previous state for next calculation."""
        self.prev_position = position.copy() if position is not None else None
        self.prev_action = action.copy() if action is not None else None
        self.prev_joint_positions = joint_pos.copy() if joint_pos is not None else None


class AdaptiveRewardCalculator(QuadrupedRewardCalculator):
    """
    Adaptive reward calculator that adjusts weights based on training progress.
    Useful for curriculum learning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_weights = RewardWeights()
        self.training_progress = 0.0  # 0.0 to 1.0
        
    def update_training_progress(self, progress: float):
        """Update training progress (0.0 = start, 1.0 = complete)."""
        self.training_progress = np.clip(progress, 0.0, 1.0)
        self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt reward weights based on training progress."""
        # Early training: focus on basic stability and movement
        if self.training_progress < 0.3:
            self.weights.stability = 0.8
            self.weights.height_maintenance = 0.6
            self.weights.forward_velocity = 0.5
            self.weights.gait_symmetry = 0.1
            
        # Mid training: introduce gait patterns
        elif self.training_progress < 0.7:
            self.weights.stability = 0.6
            self.weights.height_maintenance = 0.4
            self.weights.forward_velocity = 0.8
            self.weights.gait_symmetry = 0.4
            self.weights.leg_coordination = 0.3
            
        # Late training: focus on efficiency and natural gaits
        else:
            self.weights.stability = 0.4
            self.weights.height_maintenance = 0.3
            self.weights.forward_velocity = 1.0
            self.weights.gait_symmetry = 0.6
            self.weights.leg_coordination = 0.5
            self.weights.energy_efficiency = 0.2
            self.weights.action_smoothness = 0.2


# Convenience function for easy integration
def calculate_quadruped_reward(robot_state: Dict[str, np.ndarray], 
                             action: np.ndarray,
                             target_velocity: float = 1.0,
                             target_height: float = 0.3,
                             weights: Optional[RewardWeights] = None,
                             step_count: int = 0) -> Tuple[float, Dict[str, float]]:
    """
    Convenience function to calculate quadruped reward.
    
    Args:
        robot_state: Robot state dictionary
        action: Action taken
        target_velocity: Target forward velocity
        target_height: Target height
        weights: Reward weights
        step_count: Current step count
        
    Returns:
        Tuple of (total_reward, component_rewards)
    """
    calculator = QuadrupedRewardCalculator(
        target_velocity=target_velocity,
        target_height=target_height,
        weights=weights
    )
    
    return calculator.calculate_reward(robot_state, action, step_count=step_count)
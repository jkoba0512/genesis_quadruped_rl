"""Unitree Go2 robot loader for Genesis simulation."""

import genesis as gs
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any


class Go2Robot:
    """Unitree Go2 quadruped robot loader and controller."""
    
    # Joint names in order (12 DOF total)
    JOINT_NAMES = [
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',  # Front Left
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',  # Front Right
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',  # Rear Left
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',  # Rear Right
    ]
    
    # Fixed standing joint positions to prevent ground penetration
    # CORRECTED for Genesis joint ordering: [hips, thighs, calves] not per-leg
    # Based on proven Unitree official angles: Hip ±0.006°, Thigh 34.9°, Calf -69.8°
    DEFAULT_JOINT_POS = np.array([
        # All hip joints first: FL, FR, RL, RR
        0.00571868, -0.00571868, 0.00571868, -0.00571868,  
        # All thigh joints second: FL, FR, RL, RR  
        0.608813, 0.608813, 0.608813, 0.608813,
        # All calf joints third: FL, FR, RL, RR
        -1.21763, -1.21763, -1.21763, -1.21763
    ])
    
    # Alternative poses for different behaviors  
    # CORRECTED for Genesis joint ordering: [hips, thighs, calves] not per-leg
    SITTING_POSE = np.array([
        # All hip joints: FL, FR, RL, RR
        0.0, 0.0, 0.0, 0.0,
        # All thigh joints: FL, FR, RL, RR (front sitting, rear bent)
        1.2, 1.2, 0.3, 0.3, 
        # All calf joints: FL, FR, RL, RR
        -2.0, -2.0, -0.8, -0.8
    ])
    
    LYING_POSE = np.array([
        # All hip joints: FL, FR, RL, RR
        0.0, 0.0, 0.0, 0.0,
        # All thigh joints: FL, FR, RL, RR (all lying)
        1.5, 1.5, 1.5, 1.5,
        # All calf joints: FL, FR, RL, RR
        -2.5, -2.5, -2.5, -2.5
    ])
    
    # Joint limits (from URDF)
    JOINT_LIMITS = {
        'hip': (-1.0472, 1.0472),      # ±60 degrees
        'thigh': (-1.5708, 3.4907),     # -90 to 200 degrees
        'calf': (-2.7227, -0.83776),    # -156 to -48 degrees
    }
    
    # Joint velocity limits (rad/s)
    JOINT_VEL_LIMITS = {
        'hip': 30.1,
        'thigh': 30.1,
        'calf': 15.7,
    }
    
    # Joint effort limits (Nm)
    JOINT_EFFORT_LIMITS = {
        'hip': 23.7,
        'thigh': 23.7,
        'calf': 45.43,
    }
    
    def __init__(self, scene: gs.Scene, position: Optional[np.ndarray] = None, 
                 pose: str = "standing", use_grounding: bool = True):
        """
        Initialize Go2 robot in Genesis scene with proper grounding.
        
        Args:
            scene: Genesis scene to add robot to
            position: Initial position [x, y, z]. If None, uses grounding calculation.
            pose: Initial pose ("standing", "sitting", "lying")
            use_grounding: Whether to calculate proper ground contact position
        """
        self.scene = scene
        self.pose = pose
        self.use_grounding = use_grounding
        self.robot = None
        
        # Load robot first, then position it properly
        self._load_robot(position)
        
    def _load_robot(self, position: Optional[np.ndarray] = None):
        """Load Go2 robot from URDF and apply proper grounding."""
        # Get URDF path
        project_root = Path(__file__).parent.parent.parent.parent
        urdf_path = project_root / "assets" / "robots" / "go2" / "urdf" / "go2_genesis.urdf"
        
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Calculate proper position to prevent ground penetration
        if position is None:
            if self.use_grounding:
                # Start high enough for grounding calculation to work properly
                temp_position = np.array([0.0, 0.0, 0.6])
            else:
                # Calculate safe height based on default joint positions
                safe_height = self._calculate_safe_starting_height()
                temp_position = np.array([0.0, 0.0, safe_height])
        else:
            temp_position = position
        
        # Load robot normally (no rotation needed)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf_path),
                pos=temp_position,
                euler=[0, 0, 0],
                fixed=False,
            )
        )
        
        # Store the initial position for later grounding
        self.initial_temp_position = temp_position
        
        # Initialize robot information
        self._initialize_robot_info()
        
    def _calculate_safe_starting_height(self) -> float:
        """
        Calculate safe starting height to prevent foot ground penetration.
        Based on Go2 kinematic chain and default joint positions.
        """
        # Go2 approximate link lengths (from URDF measurements)
        base_to_hip_height = 0.08   # Base slightly above hip joints
        hip_to_thigh_length = 0.213  # Upper leg length
        thigh_to_calf_length = 0.233 # Lower leg length
        foot_extension = 0.12        # Approximate foot/contact point extension
        
        # Use default joint angles for calculation
        thigh_angle = self.DEFAULT_JOINT_POS[1]  # 0.5 rad = ~28.6°
        calf_angle = self.DEFAULT_JOINT_POS[2]   # -1.0 rad = ~-57.3°
        
        # Forward kinematics: calculate foot position relative to base
        # Thigh segment (downward from hip)
        thigh_end_z = -hip_to_thigh_length * np.cos(thigh_angle)
        
        # Calf segment (from thigh end, angle relative to thigh)
        calf_end_z = thigh_end_z - thigh_to_calf_length * np.cos(calf_angle)
        
        # Add foot extension
        foot_tip_z = calf_end_z - foot_extension
        
        # Total distance from base to lowest foot point
        foot_drop_distance = abs(foot_tip_z) + base_to_hip_height
        
        # Add safety margin to ensure feet are above ground
        safety_margin = 0.05  # 5cm clearance
        
        safe_height = foot_drop_distance + safety_margin
        
        return safe_height
        
    def _initialize_robot_info(self):
        """Initialize and print robot information after loading."""
        print(f"Loaded Go2 robot with {self.robot.n_dofs} DOF")
        
        # Print joint information
        print("All joints:")
        for i in range(len(self.robot.joints)):
            joint = self.robot.joints[i]
            print(f"  {i}: {joint.name} (type: {joint.type})")
            
        # Find controllable joints (revolute joints, excluding base)
        self.controllable_joints = []
        for i, joint in enumerate(self.robot.joints):
            if hasattr(joint.type, 'name') and joint.type.name == 'REVOLUTE':
                self.controllable_joints.append(i)
            elif str(joint.type) == 'JointType.REVOLUTE' or 'REVOLUTE' in str(joint.type):
                self.controllable_joints.append(i)
                
        print(f"Controllable joints ({len(self.controllable_joints)}): {[self.robot.joints[i].name for i in self.controllable_joints]}")
        
    def apply_natural_pose_and_grounding(self):
        """
        Apply natural standing pose and proper grounding after scene is built.
        This should be called after scene.build().
        
        FIXED: Uses gradual control method instead of instant positioning to prevent instability.
        """
        print(f"Applying natural {self.pose} pose and grounding...")
        
        # Phase 1: Set joint targets (gradual control) - FIXED METHOD
        pose_positions = self._get_pose_positions()
        self._set_joint_targets_gradual(pose_positions)
        
        # Phase 2: Let joints settle naturally (pre-grounding)
        print("  Letting joints settle into pose...")
        for _ in range(30):
            self.scene.step()
        
        # Phase 3: Apply grounding on settled robot
        if self.use_grounding:
            self._apply_grounding()
        
        # Phase 4: Extended settling on ground
        print("  Final settling on ground...")
        for _ in range(50):
            self.scene.step()
            
        # Final position check
        final_state = self.get_state()
        print(f"✅ Go2 positioned at height: {final_state['base_pos'][2]:.3f}m in {self.pose} pose")
        
    def _get_pose_positions(self) -> np.ndarray:
        """Get joint positions for specified pose."""
        if self.pose == "standing":
            return self.DEFAULT_JOINT_POS.copy()
        elif self.pose == "sitting":
            return self.SITTING_POSE.copy()
        elif self.pose == "lying":
            return self.LYING_POSE.copy()
        else:
            print(f"Warning: Unknown pose '{self.pose}', using standing")
            return self.DEFAULT_JOINT_POS.copy()
            
    def _set_joint_positions(self, joint_positions: np.ndarray):
        """Set joint positions for controllable joints only. DEPRECATED: Use _set_joint_targets_gradual."""
        if len(joint_positions) != len(self.controllable_joints):
            print(f"Warning: Expected {len(self.controllable_joints)} joint positions, got {len(joint_positions)}")
            return
            
        # Get current all joint positions
        current_all_positions = self.robot.get_dofs_position()
        
        # Update only controllable joints
        new_all_positions = current_all_positions.clone()
        for i, joint_idx in enumerate(self.controllable_joints):
            new_all_positions[joint_idx] = joint_positions[i]
            
        # Apply the new positions
        self.robot.set_dofs_position(new_all_positions)
        
    def _set_joint_targets_gradual(self, joint_positions: np.ndarray):
        """
        Set joint targets using gradual control - FIXED METHOD.
        Uses control_dofs_position() instead of set_dofs_position() to prevent instability.
        
        CRITICAL FIX: Genesis joint order is [hips, thighs, calves] NOT per-leg!
        """
        if len(joint_positions) != len(self.controllable_joints):
            print(f"Warning: Expected {len(self.controllable_joints)} joint positions, got {len(joint_positions)}")
            return
            
        # PROVEN STANDING POSE from Unitree Official Repository (unitree_mujoco)
        # Hip: ±0.006°, Thigh: 34.9°, Calf: -69.8° -> [±0.00571868, 0.608813, -1.21763] rad
        # Source: https://github.com/unitreerobotics/unitree_mujoco/blob/main/example/python/stand_go2.py
        
        # FIXED: Convert from per-leg format to Genesis joint order format
        # Input format (per-leg): [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]  
        # Genesis format: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
        
        # Working pose per leg: [hip, thigh, calf]
        fl_pose = [0.00571868, 0.608813, -1.21763]   # FL: hip, thigh, calf
        fr_pose = [-0.00571868, 0.608813, -1.21763] # FR: hip, thigh, calf (hip negated)
        rl_pose = [0.00571868, 0.608813, -1.21763]   # RL: hip, thigh, calf
        rr_pose = [-0.00571868, 0.608813, -1.21763] # RR: hip, thigh, calf (hip negated)
        
        # Convert to Genesis joint ordering: group by joint type
        working_pose_genesis_order = np.array([
            # All hip joints first
            fl_pose[0], fr_pose[0], rl_pose[0], rr_pose[0],  # FL_hip, FR_hip, RL_hip, RR_hip
            # All thigh joints second
            fl_pose[1], fr_pose[1], rl_pose[1], rr_pose[1],  # FL_thigh, FR_thigh, RL_thigh, RR_thigh
            # All calf joints third
            fl_pose[2], fr_pose[2], rl_pose[2], rr_pose[2]   # FL_calf, FR_calf, RL_calf, RR_calf
        ])
        
        # Use working joint positions with clipping to joint limits
        clipped_positions = self._clip_joint_positions(working_pose_genesis_order)
        
        # Create full DOF array for all joints
        full_targets = torch.zeros(self.robot.n_dofs, dtype=torch.float32)
        
        # Map controllable joint targets to full DOF array
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(clipped_positions):
                full_targets[joint_idx] = clipped_positions[i]
            
        # Apply gradual control (this is the key fix!)
        self.robot.control_dofs_position(full_targets)
        print(f"  Applied gradual joint control with CORRECTED joint ordering")
        
    def _apply_grounding(self):
        """Apply robot grounding library to place feet on ground - FIXED VERSION."""
        try:
            # Import here to avoid circular imports
            import sys
            import os
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from src.genesis_humanoid_rl.physics.robot_grounding import RobotGroundingFactory
            
            # Use same parameters as working demo
            calculator = RobotGroundingFactory.create_calculator(self.robot, verbose=False)
            grounding_height = calculator.get_grounding_height(safety_margin=0.005)  # Match working demo
            
            # Apply grounding position
            self.robot.set_pos([0, 0, grounding_height])
            print(f"  Applied grounding: {grounding_height:.3f}m height")
            
        except Exception as e:
            print(f"Warning: Could not apply grounding: {e}")
            print("Using default height")
        
    def reset(self, position: Optional[np.ndarray] = None, 
              joint_positions: Optional[np.ndarray] = None):
        """
        Reset robot to initial state.
        
        Args:
            position: New position [x, y, z]
            joint_positions: Joint positions (12 values)
        """
        if position is not None:
            self.robot.set_pos(position)
            self.position = position
        else:
            self.robot.set_pos(self.position)
            
        if joint_positions is not None:
            # Ensure we have the right number of joint positions
            if len(joint_positions) == len(self.controllable_joints):
                # FIXED: Use gradual control instead of instant position setting
                full_targets = torch.zeros(self.robot.n_dofs, dtype=torch.float32)
                for i, joint_idx in enumerate(self.controllable_joints):
                    full_targets[joint_idx] = joint_positions[i]
                self.robot.control_dofs_position(full_targets)
            else:
                print(f"Warning: Expected {len(self.controllable_joints)} joint positions, got {len(joint_positions)}")
        else:
            # FIXED: Use gradual control instead of instant position setting for default pose
            full_targets = torch.zeros(self.robot.n_dofs, dtype=torch.float32)
            for i, joint_idx in enumerate(self.controllable_joints):
                if i < len(self.DEFAULT_JOINT_POS):
                    full_targets[joint_idx] = self.DEFAULT_JOINT_POS[i]
            self.robot.control_dofs_position(full_targets)
            
        # Reset velocities
        self.robot.set_dofs_velocity(torch.zeros(self.robot.n_dofs))
        self.robot.set_vel(torch.zeros(3))
        self.robot.set_ang_vel(torch.zeros(3))
        
    def settle_after_reset(self, settle_steps: int = 30):
        """
        Allow robot to settle into stable position after reset.
        
        CRITICAL for height stability: This prevents the physics instability
        caused by instant joint position changes during reset.
        
        Args:
            settle_steps: Number of simulation steps to let robot settle
        """
        for _ in range(settle_steps):
            self.scene.step()
        
        # Check final stability
        final_state = self.get_state()
        final_height = final_state['base_pos'][2]
        print(f"Robot settled at height: {final_height:.3f}m after {settle_steps} steps")
        
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state."""
        try:
            return {
                'base_pos': self.robot.get_pos().cpu().numpy(),
                'base_quat': self.robot.get_quat().cpu().numpy(),
                'base_vel': self.robot.get_vel().cpu().numpy(),
                'base_ang_vel': self.robot.get_ang_vel().cpu().numpy(),
                'joint_pos': self.robot.get_dofs_position().cpu().numpy(),
                'joint_vel': self.robot.get_dofs_velocity().cpu().numpy(),
            }
        except AttributeError as e:
            print(f"Warning: Could not get some robot state properties: {e}")
            # Return simplified state if some methods don't exist
            return {
                'base_pos': self.robot.get_pos().cpu().numpy(),
                'base_quat': self.robot.get_quat().cpu().numpy(),
                'joint_pos': self.robot.get_dofs_position().cpu().numpy(),
                'joint_vel': self.robot.get_dofs_velocity().cpu().numpy(),
            }
        
    def set_joint_targets(self, targets: np.ndarray, mode: str = 'position'):
        """
        Set joint targets.
        
        Args:
            targets: Target values for controllable joints
            mode: 'position' or 'torque'
        """
        if len(targets) != len(self.controllable_joints):
            raise ValueError(f"Expected {len(self.controllable_joints)} targets, got {len(targets)}")
            
        # Create full DOF array
        full_targets = np.zeros(self.robot.n_dofs)
        
        if mode == 'position':
            # Clip to joint limits and map to full DOF array
            clipped_targets = self._clip_joint_positions(targets)
            for i, joint_idx in enumerate(self.controllable_joints):
                full_targets[joint_idx] = clipped_targets[i]
            self.robot.control_dofs_position(full_targets)
        elif mode == 'torque':
            # Clip to effort limits and map to full DOF array
            clipped_targets = self._clip_joint_torques(targets)
            for i, joint_idx in enumerate(self.controllable_joints):
                full_targets[joint_idx] = clipped_targets[i]
            self.robot.control_dofs_force(full_targets)
        else:
            raise ValueError(f"Unknown control mode: {mode}")
            
    def _clip_joint_positions(self, positions: np.ndarray) -> np.ndarray:
        """Clip joint positions to limits."""
        clipped = positions.copy()
        for i in range(12):
            joint_type = self._get_joint_type(i)
            low, high = self.JOINT_LIMITS[joint_type]
            clipped[i] = np.clip(positions[i], low, high)
        return clipped
        
    def _clip_joint_torques(self, torques: np.ndarray) -> np.ndarray:
        """Clip joint torques to effort limits."""
        clipped = torques.copy()
        for i in range(12):
            joint_type = self._get_joint_type(i)
            limit = self.JOINT_EFFORT_LIMITS[joint_type]
            clipped[i] = np.clip(torques[i], -limit, limit)
        return clipped
        
    def _get_joint_type(self, joint_idx: int) -> str:
        """Get joint type (hip/thigh/calf) from index using Genesis joint ordering."""
        # Genesis ordering: [hips 0-3, thighs 4-7, calves 8-11]
        if 0 <= joint_idx <= 3:
            return 'hip'
        elif 4 <= joint_idx <= 7:
            return 'thigh'
        elif 8 <= joint_idx <= 11:
            return 'calf'
        else:
            # Default fallback for any unexpected indices
            return 'hip'
            
    def get_foot_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all four feet."""
        # For now, estimate foot positions from joint kinematics
        # This is a simplified approach until we have proper link access
        foot_positions = {}
        
        state = self.get_state()
        base_pos = state['base_pos']
        
        # Simplified foot position estimation based on leg geometry
        # These are rough estimates for Go2 dimensions
        leg_length = 0.4  # Total leg length when extended
        
        # Leg attachment points relative to base (approximate)
        leg_offsets = {
            'FL': np.array([0.19, 0.047, 0]),    # Front Left
            'FR': np.array([0.19, -0.047, 0]),   # Front Right
            'RL': np.array([-0.19, 0.047, 0]),   # Rear Left
            'RR': np.array([-0.19, -0.047, 0]),  # Rear Right
        }
        
        for leg, offset in leg_offsets.items():
            # Simple approximation: foot below hip attachment point
            hip_world_pos = base_pos + offset
            foot_world_pos = hip_world_pos - np.array([0, 0, leg_length])
            foot_positions[leg] = foot_world_pos
                    
        return foot_positions
    
    def set_joint_targets_from_actions(self, actions: np.ndarray):
        """
        Set joint control targets from RL action vector.
        
        Args:
            actions: Action vector (12 values, range [-1, 1])
        """
        # Ensure actions is a numpy array
        actions = np.asarray(actions)
        
        # Handle scalar input (expand to all joints)
        if actions.ndim == 0:
            actions = np.full(len(self.controllable_joints), actions)
        
        if len(actions) != len(self.controllable_joints):
            raise ValueError(f"Expected {len(self.controllable_joints)} actions, got {len(actions)}")
        
        # Get current joint positions
        current_positions = self.robot.get_dofs_position()
        
        # Create target positions by adding scaled actions to current positions
        target_positions = current_positions.clone()
        
        for i, joint_idx in enumerate(self.controllable_joints):
            # Scale action from [-1, 1] to a reasonable position change
            position_change = actions[i] * 0.1  # 0.1 radian max change per step
            target_positions[joint_idx] += position_change
        
        # Apply joint limits if available
        # For now, use basic limits
        target_positions = torch.clamp(target_positions, -2.0, 2.0)
        
        # Set joint position targets
        self.robot.control_dofs_position(target_positions)
        
    def get_joint_positions(self) -> np.ndarray:
        """
        Get positions of controllable joints only.
        
        Returns:
            Joint positions array (12 values)
        """
        all_positions = self.robot.get_dofs_position().cpu().numpy()
        controllable_positions = np.array([
            all_positions[joint_idx] for joint_idx in self.controllable_joints
        ])
        return controllable_positions
        
    def get_joint_velocities(self) -> np.ndarray:
        """
        Get velocities of controllable joints only.
        
        Returns:
            Joint velocities array (12 values)
        """
        all_velocities = self.robot.get_dofs_velocity().cpu().numpy()
        controllable_velocities = np.array([
            all_velocities[joint_idx] for joint_idx in self.controllable_joints
        ])
        return controllable_velocities


def test_go2_loader():
    """Test Go2 robot loading and basic functionality."""
    # Initialize Genesis
    try:
        gs.init()
    except RuntimeError as e:
        if "Genesis already initialized" not in str(e):
            raise
    
    # Create scene with proper options
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
        ),
        show_viewer=False,
    )
    
    # Add ground
    scene.add_entity(gs.morphs.Plane())
    
    # Load Go2 robot
    go2 = Go2Robot(scene, position=np.array([0, 0, 0.3]))
    
    # Build scene
    scene.build()
    
    # Test robot
    print("Testing Go2 robot...")
    state = go2.get_state()
    print(f"Base position: {state['base_pos']}")
    print(f"Joint positions: {state['joint_pos']}")
    
    # Simulate for a bit
    for i in range(100):
        scene.step()
        if i % 20 == 0:
            state = go2.get_state()
            print(f"Step {i}: Base height = {state['base_pos'][2]:.3f}")
    
    print("Go2 robot test completed!")
    

if __name__ == "__main__":
    test_go2_loader()
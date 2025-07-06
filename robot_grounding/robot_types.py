"""
Robot type detection for proper grounding calculation.
"""

from enum import Enum
from typing import List, Dict, Any
import re


class RobotType(Enum):
    """Supported robot types."""
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped" 
    UNKNOWN = "unknown"


class RobotTypeDetector:
    """
    Detect robot type based on link structure and naming patterns.
    """
    
    # Robot type indicators
    HUMANOID_INDICATORS = {
        'names': ['g1', 'humanoid', 'bipedal'],
        'patterns': [r'.*arm.*', r'.*hand.*', r'.*leg.*', r'.*torso.*'],
        'typical_dof': range(30, 40),  # G1 has 35 DOF
        'min_links': 20
    }
    
    QUADRUPED_INDICATORS = {
        'names': ['go2', 'quadruped', 'dog', 'a1', 'anymal'],
        'patterns': [r'[FB][LR]_.*', r'.*_hip.*', r'.*_thigh.*', r'.*_calf.*'],
        'typical_dof': range(10, 20),  # Go2 has 12 controllable DOF
        'min_links': 8
    }
    
    @staticmethod
    def detect_robot_type(robot) -> RobotType:
        """
        Detect robot type based on robot structure.
        
        Args:
            robot: Genesis robot entity
            
        Returns:
            Detected robot type
        """
        # Get basic robot info
        n_links = robot.n_links
        n_dofs = robot.n_dofs
        
        # Get all link names
        link_names = []
        for link in robot.links:
            if hasattr(link, 'name'):
                link_names.append(link.name.lower())
        
        # Score each robot type
        humanoid_score = RobotTypeDetector._score_robot_type(
            link_names, n_links, n_dofs, RobotTypeDetector.HUMANOID_INDICATORS
        )
        
        quadruped_score = RobotTypeDetector._score_robot_type(
            link_names, n_links, n_dofs, RobotTypeDetector.QUADRUPED_INDICATORS
        )
        
        # Determine robot type
        if quadruped_score > humanoid_score and quadruped_score > 0.3:
            return RobotType.QUADRUPED
        elif humanoid_score > quadruped_score and humanoid_score > 0.3:
            return RobotType.HUMANOID
        else:
            return RobotType.UNKNOWN
    
    @staticmethod
    def _score_robot_type(link_names: List[str], n_links: int, n_dofs: int, 
                         indicators: Dict) -> float:
        """
        Score how well robot matches a specific type.
        
        Args:
            link_names: List of link names (lowercase)
            n_links: Number of links
            n_dofs: Number of DOF
            indicators: Type indicators dictionary
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Check name indicators (30% weight)
        for name in indicators['names']:
            for link_name in link_names:
                if name in link_name:
                    score += 0.3
                    break
        
        # Check pattern matches (40% weight)
        pattern_matches = 0
        total_patterns = len(indicators['patterns'])
        
        for pattern in indicators['patterns']:
            regex = re.compile(pattern)
            for link_name in link_names:
                if regex.match(link_name):
                    pattern_matches += 1
                    break
        
        if total_patterns > 0:
            score += 0.4 * (pattern_matches / total_patterns)
        
        # Check DOF range (20% weight)
        if n_dofs in indicators['typical_dof']:
            score += 0.2
        
        # Check link count (10% weight)
        if n_links >= indicators['min_links']:
            score += 0.1
        
        return min(score, 1.0)
    
    @staticmethod
    def get_default_grounding_config(robot_type: RobotType) -> Dict[str, Any]:
        """
        Get default grounding configuration for robot type.
        
        Args:
            robot_type: Detected robot type
            
        Returns:
            Configuration dictionary
        """
        if robot_type == RobotType.HUMANOID:
            return {
                'fallback_height': 0.78,  # G1 humanoid standing height
                'safety_margin': 0.03,
                'expected_feet': 2,
                'foot_keywords': ['foot', 'ankle', 'toe'],
            }
        elif robot_type == RobotType.QUADRUPED:
            return {
                'fallback_height': 0.30,  # Go2 quadruped standing height
                'safety_margin': 0.02,
                'expected_feet': 4,
                'foot_keywords': ['calf', 'foot', 'paw'],
            }
        else:
            return {
                'fallback_height': 0.50,  # Generic fallback
                'safety_margin': 0.03,
                'expected_feet': 2,
                'foot_keywords': ['foot', 'end_effector'],
            }
# CLAUDE.md

This file provides essential guidance to Claude Code when working with the genesis_quadruped_rl project.

## Project Overview
**Genesis Quadruped RL** - Framework for training quadruped robots to achieve stable walking patterns using reinforcement learning.

- **Repository**: https://github.com/jkoba0512/genesis_quadruped_rl
- **Status**: 🚧 In Development - adapting from genesis_humanoid_rl
- **Purpose**: Train Unitree Go2 quadruped robot (12 DOF) to walk using PPO with curriculum learning
- **Base Project**: Adapted from genesis_humanoid_rl (18 development phases completed)

## Technology Stack
- **Python**: 3.10 (required for compatibility)
- **Physics**: Genesis v0.2.1 (`pip install genesis-world`)
- **RL**: Stable-Baselines3 (`pip install stable-baselines3[extra]`)
- **Robot**: Unitree Go2 quadruped with automatic grounding system
- **Package Manager**: uv (modern Python package manager)
- **GPU**: NVIDIA with CUDA support recommended

## Quick Start Commands
```bash
# Setup (one-time)
git clone https://github.com/jkoba0512/genesis_quadruped_rl.git
cd genesis_quadruped_rl
uv run python tools/setup_environment.py

# Training
uv run python scripts/train_sb3.py --config configs/test.json          # Quick test (5 min)
uv run python scripts/train_curriculum.py --config configs/curriculum_medium.json  # Recommended (1-2 hours)

# Monitoring
uv run tensorboard --logdir ./logs/curriculum_medium --port 6006

# Evaluation & Video
uv run python scripts/evaluate_sb3.py ./models/sb3/best_model --render --episodes 5
uv run python scripts/genesis_video_record.py --steps 200
```

## Project Structure
```
genesis_quadruped_rl/
├── scripts/                   # Training, evaluation, video recording scripts
├── src/genesis_quadruped_rl/   
│   ├── environments/         # RL environments (quadruped_env.py, curriculum_env.py)
│   ├── curriculum/           # Curriculum learning system
│   ├── rewards/              # Reward functions
│   ├── domain/               # DDD architecture (value objects, entities, services)
│   ├── api/                  # REST API (46 endpoints)
│   └── infrastructure/       # Genesis adapter, monitoring, error handling
├── configs/                   # JSON training configurations
├── assets/robots/go2/        # Unitree Go2 URDF and meshes
└── tools/                    # Setup and verification scripts
```

## Key Features
1. **Curriculum Learning**: Progressive training stages (Balance → Trot → Gallop → Advanced)
2. **Parallel Training**: Multi-environment support with automatic resource management
3. **Video Recording**: Genesis camera integration for training visualization (1920x1080 MP4)
4. **REST API**: 46 endpoints for training management and monitoring
5. **DDD Architecture**: Enterprise-grade domain-driven design with clean separation
6. **Automatic Grounding**: Robot positioning system preventing ground penetration
7. **Advanced Testing**: 300+ tests with fixtures and context managers
8. **Database Migration**: Optimized schema with 30% performance improvement

## Critical Genesis v0.2.1 Compatibility Notes

### API Changes (Must Know!)
```python
# Scene Creation - REMOVED solver parameter
gs.init()  # Required before scene creation!
scene = gs.Scene(
    sim_options=gs.SimOptions(
        dt=0.01,
        substeps=2  # Use substeps, not solver
    )
)

# Entity Addition - NO material parameter
scene.add_entity(gs.morphs.Plane())  # Direct addition

# Robot DOF Access
robot.n_dofs  # Not len(robot.dofs)
robot.get_dofs_position()  # Not robot.dofs_position
robot.get_dofs_velocity()  # Not robot.dofs_velocity

# No scene.close() - Genesis handles cleanup
```

### Common Issues & Fixes
1. **"Genesis hasn't been initialized"**: Call `gs.init()` before scene creation
2. **SB3 activation_fn error**: Remove `"activation_fn": "tanh"` from policy_kwargs
3. **TensorBoard remote access**: Use `--host 0.0.0.0` for VPN access
4. **Missing trimesh**: Run `uv add trimesh` (required for mesh loading)

## Training Configurations
- `test.json` - Quick 5k steps test
- `default.json` - Production 1M steps  
- `curriculum_test.json` - Quick curriculum (10k steps)
- `curriculum_medium.json` - Recommended (100k steps, 1-2 hours)
- `curriculum.json` - Full training (2M steps)
- `progressive_phase[1-3].json` - Multi-phase progressive training

## Reward System
The robot learns to walk through these reward components:
- **Forward Velocity** (1.0x): Movement toward target speed
- **Base Stability** (0.5x): Stable body orientation (pitch/roll limits)
- **Height** (0.3x): Proper walking height (~0.3m for Go2)
- **Energy Efficiency** (-0.1x): Smooth, efficient motion
- **Action Smoothness** (-0.1x): Continuous movements
- **Foot Contact** (0.4x): Proper gait patterns (trot, pace, bound)
- **Base Angular Velocity** (-0.2x): Minimizes unwanted rotation
- **Symmetry** (0.2x): Encourages symmetric leg movements

## Development Workflow

### Adding New Features
1. Check existing patterns in similar files
2. Follow DDD architecture (domain → application → infrastructure)
3. Update tests in corresponding test files
4. Use existing reward components as templates

### Training a Model
```bash
# 1. Choose configuration
# 2. Start training with monitoring
uv run python scripts/train_curriculum.py --config configs/curriculum_medium.json
uv run tensorboard --logdir ./logs/curriculum_medium

# 3. Generate videos at checkpoints
uv run python scripts/checkpoint_video_generator.py

# 4. Evaluate final model
uv run python scripts/evaluate_sb3.py ./models/curriculum_medium/best_model --render
```

### API Usage
```bash
# Start API server
uv run python -m genesis_quadruped_rl.api.cli --dev --port 8001

# Access endpoints
curl http://localhost:8001/health/
curl http://localhost:8001/training/sessions
curl http://localhost:8001/robots/
```

## Quadruped-Specific Considerations
- **Gait Patterns**: Trot, pace, bound, and gallop gaits
- **Foot Contact Sequences**: Diagonal pairs for trot, lateral pairs for pace
- **Lower CoM**: ~0.3m compared to humanoid's ~0.8m
- **4-leg Stability**: More stable than bipedal, allows faster learning
- **12 DOF**: 3 per leg (hip abduction, hip flexion, knee flexion)
- **Observation Space**: ~50 dimensions (reduced from humanoid's 113)
- **Action Space**: 12 continuous joint controls (vs 35 for humanoid)

## Performance Expectations
- **Simulation**: 200+ FPS with Genesis v0.2.1
- **Training**: ~150k steps/hour (varies by hardware)
- **Convergence**: 2-3x faster than humanoid due to inherent stability
- **Memory**: ~15GB for multi-environment training
- **GPU**: NVIDIA RTX 3060 Ti or better recommended

## Architecture Patterns
- **Domain-Driven Design**: Rich domain model with business logic
- **Repository Pattern**: Abstract data access interfaces
- **Unit of Work**: Transaction management
- **Anti-Corruption Layer**: Genesis adapter for clean integration
- **Command Pattern**: Application layer commands
- **Event-Driven**: Domain events for loose coupling

## Testing
```bash
# Run all tests
uv run python -m pytest tests/ -v

# Specific categories
uv run python -m pytest tests/domain/ -v        # Domain logic
uv run python -m pytest tests/infrastructure/ -v # Genesis integration
uv run python -m pytest tests/api/ -v           # REST API
```

## Additional Resources
- **Unitree Go2 Specs**: 12 DOF, 15kg weight, 0.4m height
- **API Documentation**: Auto-generated at http://localhost:8001/docs
- **TensorBoard Guide**: Monitor training at http://localhost:6006
- **GitHub Issues**: Report problems at https://github.com/jkoba0512/genesis_quadruped_rl/issues
- **Base Project Docs**: See docs/ directory for detailed architecture and design

## Current Status
🚧 Adapting humanoid codebase for quadruped robot  
🚧 Updating environments for 4-legged locomotion  
🚧 Implementing quadruped-specific reward functions  
🚧 Adjusting curriculum for gait patterns  
⏳ REST API adaptation pending  
⏳ Comprehensive testing pending  
⏳ Production deployment pending

## Migration Tasks from Humanoid to Quadruped
1. **Robot Model**: Replace G1 (35 DOF) with Go2 (12 DOF) URDF/meshes
2. **Environment**: Reduce observation space, adjust action space
3. **Rewards**: Add gait-specific rewards, adjust height targets
4. **Curriculum**: Replace walking stages with gait progression
5. **Configurations**: Update training configs for quadruped dynamics
6. **Tests**: Adapt test fixtures for 4-legged robot
7. **Database**: Update schema for quadruped-specific data
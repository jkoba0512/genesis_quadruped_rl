# Genesis Quadruped RL Documentation

This directory contains comprehensive documentation for the Genesis Quadruped RL project.

## 🎯 Current Training Status

**Active Training Progress (as of latest update):**
- 🏃 **Status**: Training in progress using chunk-based GPU isolation method
- 📊 **Episodes Completed**: 250+ episodes (5 chunks × 50 episodes)
- 🎥 **Video Chunks**: 5 progression videos recorded showcasing learning evolution
- 🎯 **Target**: 5,000 total episodes for comprehensive training
- 🖥️ **Hardware**: NVIDIA RTX 3060 Ti (8GB) with memory-safe chunking
- 🚀 **Performance**: Significant improvements with reward function redesign and extended grace period

## 📚 Documentation Structure

### Getting Started
- **[../README.md](../README.md)**: Main project overview and quick start
- **[installation.md](installation.md)**: Detailed installation instructions
- **[quickstart.md](quickstart.md)**: Step-by-step beginner guide

### Training & Configuration
- **[TRAINING_PROGRESS.md](TRAINING_PROGRESS.md)**: 🆕 Current training status & metrics
- **[VIDEO_GENERATION_GUIDE.md](VIDEO_GENERATION_GUIDE.md)**: 🎬 Complete video generation guide
- **[stable_baselines3_guide.md](stable_baselines3_guide.md)**: Complete SB3 training guide
- **[EXTENDED_TRAINING_PLAN.md](EXTENDED_TRAINING_PLAN.md)**: Comprehensive training roadmap
- **[../12_HOUR_TRAINING_GUIDE.md](../12_HOUR_TRAINING_GUIDE.md)**: 12-hour training protocol
- **[../configs/README.md](../configs/README.md)**: Configuration documentation
- **[training_guide.md](training_guide.md)**: Advanced training techniques

### Technical Documentation
- **[environment_api.md](environment_api.md)**: Environment API reference
- **[architecture.md](architecture.md)**: Project architecture overview
- **[reward_function.md](reward_function.md)**: Reward function design
- **[testing_infrastructure.md](testing_infrastructure.md)**: Testing framework documentation
- **[database_migration_guide.md](database_migration_guide.md)**: Database optimization guide

### Research & Analysis
- **[reward_function_analysis.png](reward_function_analysis.png)**: Reward function analysis
- **[DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md)**: Project evolution timeline
- **[performance_benchmarks.md](performance_benchmarks.md)**: Performance results
- **[research_applications.md](research_applications.md)**: Research use cases

### Troubleshooting & Support
- **[troubleshooting.md](troubleshooting.md)**: Common issues and solutions
- **[faq.md](faq.md)**: Frequently asked questions
- **[contributing.md](contributing.md)**: Contribution guidelines

## 🚀 Quick Navigation

| What you want to do | Documentation |
|---------------------|---------------|
| Check training progress | [TRAINING_PROGRESS.md](TRAINING_PROGRESS.md) 🆕 |
| Generate videos from chunks | [VIDEO_GENERATION_GUIDE.md](VIDEO_GENERATION_GUIDE.md) 🎬 |
| Get started quickly | [../README.md](../README.md) |
| Run 12-hour training | [../12_HOUR_TRAINING_GUIDE.md](../12_HOUR_TRAINING_GUIDE.md) |
| Continue chunk training | `./run_5k_with_isolation.sh` script |
| Understand SB3 training | [stable_baselines3_guide.md](stable_baselines3_guide.md) |
| Configure training | [../configs/README.md](../configs/README.md) |
| Fix issues | [troubleshooting.md](troubleshooting.md) |
| Understand the code | [architecture.md](architecture.md) |
| Contribute to project | [contributing.md](contributing.md) |

## 📖 Reading Order

For new users, we recommend this reading order:

1. **[../README.md](../README.md)** - Project overview
2. **[installation.md](installation.md)** - Setup your environment  
3. **[quickstart.md](quickstart.md)** - Run your first training
4. **[stable_baselines3_guide.md](stable_baselines3_guide.md)** - Deep dive into training
5. **[training_guide.md](training_guide.md)** - Advanced techniques

## 🔗 External Resources

- **[Genesis Documentation](https://genesis-world.readthedocs.io)**: Physics engine docs
- **[Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io)**: RL algorithm docs
- **[Unitree Go2 Specs](https://www.unitree.com/go2)**: Robot specifications
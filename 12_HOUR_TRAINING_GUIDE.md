# 12-Hour Go2 Quadruped Training Guide

## Overview

This guide provides everything you need to run a comprehensive 12-hour training session for the Go2 quadruped robot, with automatic video generation to observe learning progress.

## Quick Start

### 1. Easy Launch (Recommended)
```bash
uv run python start_12hour_training.py
```

This launcher will:
- ✅ Check system requirements
- ✅ Validate setup
- ✅ Start 12-hour training
- ✅ Generate videos automatically
- ✅ Create final progression video

### 2. Manual Steps

#### Test Setup First
```bash
uv run python scripts/test_12hour_setup.py
```

#### Run Full Training
```bash
uv run python scripts/train_12_hours_with_videos.py
```

#### Create Progression Video
```bash
uv run python scripts/create_progression_video.py
```

## Training Details

### Timeline
- **Duration**: ~12 hours
- **Total Steps**: 1,800,000 timesteps
- **Checkpoints**: Every 300,000 steps (2 hours)
- **Videos**: Generated at each checkpoint
- **Final Video**: Complete progression comparison

### Performance Expectations
- **Speed**: ~150,000 steps/hour
- **Memory**: ~15GB RAM usage
- **GPU**: NVIDIA recommended (4GB+ VRAM)
- **Disk**: ~10GB free space needed

### Output Files

#### Models
```
./models/go2_12hour_training/
├── checkpoints/
│   ├── go2_12hour_300000_steps.zip
│   ├── go2_12hour_600000_steps.zip
│   ├── go2_12hour_900000_steps.zip
│   ├── go2_12hour_1200000_steps.zip
│   ├── go2_12hour_1500000_steps.zip
│   └── go2_12hour_1800000_steps.zip
├── final_model.zip
└── training_config.json
```

#### Videos
```
./videos/12hour_training/
├── go2_progress_00000000_untrained.mp4
├── go2_progress_00300000_checkpoint_0.mp4
├── go2_progress_00600000_checkpoint_1.mp4
├── go2_progress_00900000_checkpoint_2.mp4
├── go2_progress_01200000_checkpoint_3.mp4
├── go2_progress_01500000_checkpoint_4.mp4
└── go2_progress_01800000_checkpoint_5.mp4
```

#### Final Progression Video
```
./videos/go2_12hour_progression.mp4
```

## Monitoring Progress

### TensorBoard
```bash
uv run tensorboard --logdir ./logs/go2_12hour_training --port 6006
```
Then open: http://localhost:6006

### Live Progress
The training script shows:
- Current step count
- Elapsed time
- Steps per hour
- Estimated completion time
- Videos created

### Example Progress Output
```
📊 Training Progress:
   Steps: 450,000
   Time: 3.2 hours
   Speed: 140,625 steps/hour
   Videos created: 2
   Estimated completion: 23:45:12
```

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 5GB disk space
- CUDA-capable GPU (optional)

### Recommended
- Python 3.10+
- 16GB+ RAM
- 10GB+ disk space
- NVIDIA GPU with 4GB+ VRAM

### Required Packages
```bash
uv add genesis-world
uv add stable-baselines3[extra]
uv add opencv-python
uv add tensorboard
uv add GPUtil
uv add psutil
```

## Troubleshooting

### Common Issues

#### Training Stops Early
- Check disk space
- Check memory usage
- Check GPU memory
- Resume with `--resume` flag

#### Video Generation Fails
- Ensure OpenCV is installed
- Check video directory permissions
- Verify Genesis rendering works

#### Low Training Speed
- Check GPU availability
- Reduce batch size if memory issues
- Close other applications
- Check thermal throttling

### Resume Training
```bash
uv run python start_12hour_training.py --resume
```

### Check Setup
```bash
uv run python start_12hour_training.py --test-only
```

### GPU Check
```bash
uv run python start_12hour_training.py --gpu-check
```

## Configuration

### Training Parameters
```json
{
  "algorithm": {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "policy_kwargs": {
      "net_arch": [256, 256, 128]
    }
  },
  "training": {
    "total_timesteps": 1800000,
    "save_freq": 300000
  }
}
```

### Video Settings
- **Frequency**: Every 300,000 steps
- **Length**: 200 steps per video
- **Resolution**: 640x480 @ 30fps
- **Format**: MP4

## Expected Results

### Learning Progression
1. **Untrained** (0 steps): Random movements, falling
2. **Early** (300k steps): Basic balance, limited walking
3. **Intermediate** (900k steps): Stable walking, improved gait
4. **Advanced** (1.5M steps): Smooth locomotion, good distance
5. **Final** (1.8M steps): Optimal walking pattern, high stability

### Performance Metrics
- **Initial**: ~0.1 average reward
- **Final**: ~0.6+ average reward
- **Distance**: 3-5 meters in 200 steps
- **Stability**: Consistent height maintenance

## Advanced Usage

### Custom Training Duration
Edit `scripts/train_12_hours_with_videos.py`:
```python
"total_timesteps": 3600000,  # 24 hours
"save_freq": 600000,         # Every 4 hours
```

### Custom Video Frequency
```python
video_callback = LongTrainingVideoCallback(
    video_freq=150000,  # Every hour
    video_length=400,   # Longer videos
)
```

### Multi-GPU Training
```python
model = PPO(
    "MlpPolicy",
    env,
    device="cuda:0",  # Specify GPU
    # ... other params
)
```

## File Structure

```
genesis_quadruped_rl/
├── start_12hour_training.py           # Main launcher
├── scripts/
│   ├── train_12_hours_with_videos.py  # Training script
│   ├── test_12hour_setup.py           # Setup validation
│   └── create_progression_video.py    # Video compilation
├── models/go2_12hour_training/        # Saved models
├── logs/go2_12hour_training/          # TensorBoard logs
├── videos/12hour_training/            # Progress videos
└── videos/go2_12hour_progression.mp4  # Final video
```

## Support

If you encounter issues:
1. Run setup test: `uv run python scripts/test_12hour_setup.py`
2. Check system requirements
3. Review troubleshooting section
4. Check logs in `./logs/go2_12hour_training/`

## Tips for Success

1. **Start Fresh**: Clear previous training data if needed
2. **Monitor Resources**: Watch RAM/GPU usage during training
3. **Plan Timing**: Start when you can monitor first few hours
4. **Backup**: Training state is saved automatically
5. **Be Patient**: Full training takes ~12 hours

---

**Ready to start?** Run: `uv run python start_12hour_training.py`
# Video Generation Guide

This guide covers all methods for generating videos from your trained Go2 quadruped models.

## 🎬 Quick Start

### Generate Video from Latest Training
```bash
# Best method - stable and reliable
uv run python stable_video_generator.py
```

### Generate Video from Specific Chunk
```bash
# Method 1: Edit stable_video_generator.py (line ~25)
# Change: model_path = Path("training_5k_restarts/models/chunk_000250.zip")
uv run python stable_video_generator.py

# Method 2: Use generate_chunk5_video.py
uv run python generate_chunk5_video.py --model training_5k_restarts/models/chunk_000250.zip --output chunk5_demo.mp4 --steps 400
```

## 📊 Available Video Generation Methods

| Script | Reliability | Initial Position | Use Case |
|--------|-------------|------------------|----------|
| 🥇 `stable_video_generator.py` | Excellent | ✅ Origin | **Recommended for all use cases** |
| 🥈 `generate_chunk5_video.py` | Good | ✅ Origin | Specific chunk evaluation |
| 🥉 `ultimate_video_fix.py` | Moderate | ⚠️ May offset | Environment testing |
| ❌ `scripts/genesis_video_record.py` | Poor | ❌ Wrong obs | Legacy only |

## 🔧 Method Details

### 1. Stable Video Generator (Recommended)

**File**: `stable_video_generator.py`

**Advantages**:
- Most reliable Genesis camera handling
- Correct robot positioning at origin
- No environment reset issues
- Stable recording state

**Usage**:
```bash
# Default (uses latest model)
uv run python stable_video_generator.py

# Custom model (edit script first)
# Line ~25: model_path = Path("training_5k_restarts/models/chunk_000100.zip")
uv run python stable_video_generator.py
```

**Output**: `go2_stable_demo.mp4`

### 2. Generate Chunk5 Video

**File**: `generate_chunk5_video.py`

**Advantages**:
- Command-line arguments
- Correct observation space (44 dimensions)
- Good for batch generation

**Usage**:
```bash
# Latest model
uv run python generate_chunk5_video.py --steps 400 --output chunk5_demo.mp4

# Specific chunk
uv run python generate_chunk5_video.py \
    --model training_5k_restarts/models/chunk_000200.zip \
    --output chunk4_demo.mp4 \
    --steps 300
```

### 3. Ultimate Video Fix

**File**: `ultimate_video_fix.py`

**Advantages**:
- Uses training environment
- Tests environment integration

**Disadvantages**:
- May have recording state issues
- Robot position might be offset

**Usage**:
```bash
uv run python ultimate_video_fix.py
```

**Output**: `go2_ultimate_fixed_walking.mp4`

### 4. Create Progression Videos

**File**: `create_chunk_progression_video.py`

**Usage**:
```bash
# Creates videos for all available chunks
uv run python create_chunk_progression_video.py
```

**Output**: Individual chunk videos + combined progression video

## 📁 Available Models

Check what chunk models you have:
```bash
ls -la training_5k_restarts/models/chunk_*.zip
```

Typical output:
```
chunk_000050.zip    # Chunk 1 (50 episodes)
chunk_000100.zip    # Chunk 2 (100 episodes) 
chunk_000200.zip    # Chunk 4 (200 episodes)
chunk_000250.zip    # Chunk 5 (250 episodes)
latest_model.zip    # Most recent (same as latest chunk)
```

## 🎯 Video Specifications

### Default Settings
- **Resolution**: 1280x720 (HD)
- **Frame Rate**: 30 FPS
- **Duration**: ~13 seconds (400 steps)
- **Format**: MP4 (H.264)
- **File Size**: 0.2-0.5 MB

### Camera Settings
```python
camera = scene.add_camera(
    res=(1280, 720),
    pos=(3.0, -2.5, 1.5),    # Camera position
    lookat=(0.0, 0.0, 0.3),  # Looking at robot
    fov=45,                  # Field of view
    GUI=False                # No GUI window
)
```

### Robot Initial Position
```python
robot.set_pos([0, 0, 0.27])  # Origin with proper Go2 height
```

## 📈 Expected Results by Training Progress

### Chunk 1 (50 episodes)
- **Behavior**: Random movements, discovering joint limits
- **Movement**: <0.2m, mostly falling/struggling
- **Duration**: Episodes end early (~100-200 steps)

### Chunk 2 (100 episodes)  
- **Behavior**: Basic coordination emerging
- **Movement**: 0.2-0.5m, some forward motion
- **Duration**: Mix of early and full episodes

### Chunk 3 (150 episodes)
- **Behavior**: Improved stability, walking attempts
- **Movement**: 0.5-1.0m, sustained movement
- **Duration**: More full episodes (400 steps)

### Chunk 4 (200 episodes)
- **Behavior**: Better gait patterns
- **Movement**: 1.0-1.5m, consistent walking
- **Duration**: Mostly full episodes

### Chunk 5 (250 episodes) - MAJOR IMPROVEMENT ✅
- **Behavior**: **Reward redesign success** - eliminated laying down, proper walking achieved
- **Movement**: 2.5+ meters, sustained upright walking
- **Duration**: Consistently full episodes
- **Key Change**: Exponential height reward prevents crawling behavior

### Chunk 6+ (300+ episodes)
- **Behavior**: Optimized gait patterns, efficient locomotion
- **Movement**: 3.0+ meters, smooth acceleration
- **Duration**: Full episodes with minimal terminations

## 🔧 Troubleshooting

### "Recording not started" Error
```bash
# Use the stable video generator instead
uv run python stable_video_generator.py
```

### "Unexpected observation shape" Error
```bash
# The script expects 44 dimensions, not 46
# Use generate_chunk5_video.py or stable_video_generator.py
```

### Video File Not Created
1. Check disk space: `df -h`
2. Check model exists: `ls training_5k_restarts/models/`
3. Check permissions: `ls -la *.mp4`
4. Try stable video generator

### Robot Starts at Wrong Position
- ✅ Use `stable_video_generator.py` - starts at origin
- ❌ Avoid `ultimate_video_fix.py` - may have offset

### Genesis Initialization Error
```bash
# If Genesis is already initialized, restart terminal or:
export GENESIS_RESET=1
uv run python stable_video_generator.py
```

## 🎬 Viewing Videos

### Local Viewing
```bash
# VLC (recommended)
vlc go2_stable_demo.mp4

# MPV
mpv go2_stable_demo.mp4

# Default player
xdg-open go2_stable_demo.mp4
```

### Web Browser Viewing
```bash
# Start HTTP server
python -m http.server 8000

# Open browser to http://localhost:8000
# Click on video files to play
```

### Copy to Shared Location
```bash
# Copy to main project directory
cp go2_stable_demo.mp4 ~/SynologyDrive/genesis_quadruped_rl/

# Copy all videos
cp *.mp4 ~/SynologyDrive/genesis_quadruped_rl/videos/
```

## 📊 Video Analysis

### Measuring Performance
```python
# The scripts automatically report:
print(f"Displacement: {total_displacement:.3f}m")
print(f"Start: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
print(f"End: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
```

### Performance Benchmarks
- **Excellent**: >2.0m displacement
- **Good**: 1.0-2.0m displacement  
- **Learning**: 0.5-1.0m displacement
- **Early**: <0.5m displacement

### What to Look For
1. **Initial Position**: Should start near `[0, 0, 0.27]`
2. **Stability**: Robot maintains upright posture
3. **Gait Pattern**: Coordinated leg movements
4. **Forward Progress**: Consistent movement in +X direction
5. **Episode Length**: Full 400-step episodes indicate stability

## 🚀 Advanced Usage

### Batch Video Generation
```bash
# Generate videos for all chunks
for chunk in training_5k_restarts/models/chunk_*.zip; do
    echo "Generating video for $chunk"
    # Edit stable_video_generator.py to use $chunk
    uv run python stable_video_generator.py
    mv go2_stable_demo.mp4 "${chunk%.zip}_demo.mp4"
done
```

### Custom Video Settings
Edit `stable_video_generator.py`:
```python
# Longer video
for step in range(800):  # 26 seconds instead of 13

# Higher resolution  
camera = scene.add_camera(res=(1920, 1080), ...)

# Different camera angle
camera = scene.add_camera(pos=(0, -4, 2), lookat=(0, 0, 0.3), ...)
```

### Multiple Camera Angles
```python
# Add multiple cameras
cam_front = scene.add_camera(res=(1280, 720), pos=(3, -2.5, 1.5), ...)
cam_side = scene.add_camera(res=(1280, 720), pos=(-4, 0, 1.5), ...)
cam_top = scene.add_camera(res=(1280, 720), pos=(0, 0, 3), ...)

# Record from different angles
cam_front.start_recording()
# ... simulation loop with cam_front.render()
cam_front.stop_recording(save_to_filename="front_view.mp4")
```

## 📝 Summary

For most use cases, use:
```bash
uv run python stable_video_generator.py
```

This provides the most reliable video generation with correct robot positioning and stable Genesis camera recording.
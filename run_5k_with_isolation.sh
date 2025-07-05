#!/bin/bash
# GPU Training with Process Isolation - Full 5K Episodes
# Runs training in 50-episode chunks with automatic restarts

echo "🚀 Starting 5K Episode Training with Process Isolation"
echo "================================================================"
echo "📊 Strategy: 50-episode chunks with process restarts"
echo "🛡️  Memory protection: Full GPU memory cleanup between chunks"
echo "⚡ GPU acceleration: Maintained throughout training"
echo ""

# Configuration
CHUNK_SIZE=50
TOTAL_EPISODES=5000
CHUNKS=$((TOTAL_EPISODES / CHUNK_SIZE))

echo "📋 Training Configuration:"
echo "   🎯 Total episodes: ${TOTAL_EPISODES:,}"
echo "   📦 Chunk size: ${CHUNK_SIZE} episodes"
echo "   🔄 Total chunks: ${CHUNKS}"
echo "   ⏱️  Estimated time: ~5 hours (1 hour per chunk)"
echo ""

# Create logs directory
mkdir -p training_5k_restarts/logs
mkdir -p training_5k_restarts/models

# Check if resuming from previous training
RESUME_FROM=0

# First check for total progress file (survives restarts)
if [ -f "training_5k_restarts/total_progress.txt" ]; then
    COMPLETED_EPISODES=$(cat training_5k_restarts/total_progress.txt)
    RESUME_FROM=$((COMPLETED_EPISODES / CHUNK_SIZE * CHUNK_SIZE))
    echo "🔄 Resuming from episode ${RESUME_FROM} (based on saved progress)"
    echo "   📈 Total episodes completed: ${COMPLETED_EPISODES}"
    echo "   📦 Chunks completed: $((COMPLETED_EPISODES / CHUNK_SIZE))"
    echo ""
elif [ -f "training_5k_restarts/logs/monitor.csv" ]; then
    # Fallback to counting from monitor.csv
    EXISTING_EPISODES=$(($(wc -l < training_5k_restarts/logs/monitor.csv) - 2))
    if [ $EXISTING_EPISODES -gt 0 ]; then
        RESUME_FROM=$((EXISTING_EPISODES / CHUNK_SIZE * CHUNK_SIZE))
        echo "🔄 Resuming from episode ${RESUME_FROM}"
        echo "   📈 Previous progress: ${EXISTING_EPISODES} episodes completed"
        echo ""
    fi
fi

# Training loop with process isolation
START_CHUNK=$((RESUME_FROM / CHUNK_SIZE + 1))

for chunk in $(seq $START_CHUNK $CHUNKS); do
    EPISODE_START=$(((chunk - 1) * CHUNK_SIZE))
    
    echo "=========================================="
    echo "🎬 CHUNK ${chunk}/${CHUNKS}: Episodes ${EPISODE_START} → $((EPISODE_START + CHUNK_SIZE))"
    echo "=========================================="
    echo "📅 Started: $(date)"
    echo ""
    
    # Run training chunk
    uv run python train_with_restarts.py \
        --start_episode $EPISODE_START \
        --num_episodes $CHUNK_SIZE \
        --total_target $TOTAL_EPISODES
    
    CHUNK_EXIT_CODE=$?
    
    echo ""
    echo "📊 Chunk ${chunk} completed with exit code: ${CHUNK_EXIT_CODE}"
    echo "📅 Finished: $(date)"
    
    # Check if chunk completed successfully
    if [ $CHUNK_EXIT_CODE -ne 0 ]; then
        echo "❌ Chunk ${chunk} failed with exit code ${CHUNK_EXIT_CODE}"
        echo "💾 Progress saved - can resume manually"
        echo "🔄 Resume command: ./run_100k_with_isolation.sh"
        exit 1
    fi
    
    # Memory cleanup between chunks
    echo "🧹 Cleaning up memory between chunks..."
    sleep 10  # Allow complete memory cleanup
    
    # Progress report
    TOTAL_COMPLETED=$((chunk * CHUNK_SIZE))
    PROGRESS_PERCENT=$((TOTAL_COMPLETED * 100 / TOTAL_EPISODES))
    
    echo ""
    echo "📈 OVERALL PROGRESS: ${TOTAL_COMPLETED:,}/${TOTAL_EPISODES:,} episodes (${PROGRESS_PERCENT}%)"
    
    # Estimated time remaining
    if [ $chunk -gt 1 ]; then
        CHUNKS_REMAINING=$((CHUNKS - chunk))
        echo "⏱️  Estimated chunks remaining: ${CHUNKS_REMAINING} (~${CHUNKS_REMAINING} hours)"
    fi
    
    echo ""
    
    # Safety check - verify model was saved
    if [ ! -f "training_5k_restarts/models/latest_model.zip" ]; then
        echo "⚠️  Warning: latest_model.zip not found after chunk ${chunk}"
        echo "💾 Checking for emergency saves..."
        ls -la training_5k_restarts/models/emergency_* 2>/dev/null || echo "   No emergency saves found"
    fi
    
done

echo ""
echo "🏆 5K TRAINING COMPLETED SUCCESSFULLY!"
echo "================================================================"
echo "📊 Final Statistics:"
echo "   🎯 Episodes: ${TOTAL_EPISODES:,}"
echo "   📦 Chunks: ${CHUNKS}"
echo "   📅 Completed: $(date)"
echo ""
echo "🎬 Next Steps:"
echo "   1. Generate training videos: python generate_training_videos.py"
echo "   2. Evaluate final model: python evaluate_trained_go2.py"
echo "   3. Create comparison videos: python create_progression_video.py"
echo ""
echo "💾 Model Location: training_5k_restarts/models/latest_model.zip"
echo "📊 Logs Location: training_5k_restarts/logs/"
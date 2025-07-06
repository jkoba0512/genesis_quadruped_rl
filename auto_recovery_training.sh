#!/bin/bash
# Auto-Recovery Training Script with GPU Memory Management
# Automatically restarts training when GPU memory issues occur

echo "🤖 Auto-Recovery Training - GPU Memory Safe"
echo "=============================================="

cd ~/PrivateNAS/genesis_quadruped_rl

# Configuration
MAX_RETRIES=50
RETRY_COUNT=0
TARGET_EPISODES=5000

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo ""
    echo "🔄 Training Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    echo "$(date): Starting training session..."
    
    # Check current progress
    # First check total_progress.txt (survives restarts)
    if [ -f "training_5k_restarts/total_progress.txt" ]; then
        CURRENT_EPISODES=$(cat training_5k_restarts/total_progress.txt)
        echo "📊 Using saved progress from total_progress.txt: $CURRENT_EPISODES episodes"
    elif [ -f "training_5k_restarts/logs/monitor.csv" ]; then
        CURRENT_EPISODES=$(($(wc -l < training_5k_restarts/logs/monitor.csv) - 2))
        if [ $CURRENT_EPISODES -lt 0 ]; then
            CURRENT_EPISODES=0
        fi
        echo "📊 Using monitor.csv progress: $CURRENT_EPISODES episodes"
    else
        CURRENT_EPISODES=0
        echo "📊 Starting fresh: 0 episodes"
    fi
    
    # Check if training is complete
    if [ $CURRENT_EPISODES -ge $TARGET_EPISODES ]; then
        echo "🎉 Training completed! $CURRENT_EPISODES episodes finished."
        break
    fi
    
    # Calculate next chunk start
    CHUNK_START=$((CURRENT_EPISODES / 50 * 50))
    if [ $CURRENT_EPISODES -gt $CHUNK_START ]; then
        CHUNK_START=$((CHUNK_START + 50))
    fi
    
    echo "📊 Current progress: $CURRENT_EPISODES episodes"
    echo "🚀 Starting from episode: $CHUNK_START"
    
    # Clear GPU memory before starting
    echo "🧹 Clearing GPU memory..."
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9
    sleep 5
    
    # Start training with timeout and error capture
    timeout 3600 uv run python train_with_restarts.py \
        --start_episode $CHUNK_START \
        --num_episodes 50 \
        --total_target $TARGET_EPISODES 2>&1 | tee -a auto_recovery.log
    
    EXIT_CODE=$?
    
    echo "📊 Training session ended with exit code: $EXIT_CODE"
    
    # Check exit reason
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Chunk completed successfully!"
        RETRY_COUNT=0  # Reset retry count on success
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "⏰ Training timed out (1 hour) - will restart"
        RETRY_COUNT=$((RETRY_COUNT + 1))
    else
        echo "❌ Training failed - checking reason..."
        
        # Check for GPU memory error
        if tail -20 auto_recovery.log | grep -q "CUDA_ERROR_OUT_OF_MEMORY\|out of memory"; then
            echo "🧠 GPU memory error detected - clearing and retrying"
            
            # Aggressive GPU cleanup
            nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9
            sleep 10
            
            # Check GPU memory
            GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            echo "🖥️  GPU Memory after cleanup: ${GPU_USED}MB"
            
            if [ $GPU_USED -gt 1000 ]; then
                echo "⚠️  GPU memory still high, waiting longer..."
                sleep 30
            fi
            
        else
            echo "🔍 Unknown error - checking logs"
            tail -10 auto_recovery.log
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
    
    # Wait between retries
    if [ $RETRY_COUNT -gt 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "⏱️  Waiting 30 seconds before retry..."
        sleep 30
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "❌ Maximum retries reached. Training stopped."
    echo "📊 Final progress: $CURRENT_EPISODES/$TARGET_EPISODES episodes"
else
    echo "🎉 Training completed successfully!"
    echo "📊 Final progress: $CURRENT_EPISODES/$TARGET_EPISODES episodes"
fi

echo "📝 Full log available in: auto_recovery.log"
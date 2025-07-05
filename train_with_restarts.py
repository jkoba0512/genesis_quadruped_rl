#!/usr/bin/env python3
"""
GPU Training with Process Isolation
==================================

Trains in 50-episode chunks with process restarts to prevent Genesis memory accumulation.
Supports resuming from any episode and maintains training continuity.
"""

import sys
import os
import time
import gc
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from genesis_quadruped_rl.environments.quadruped_env import QuadrupedWalkingEnv


def get_gpu_memory_info():
    """Get GPU memory information."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        cached = torch.cuda.memory_reserved(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        free = total - max(allocated, cached)
        return {
            'allocated': allocated,
            'cached': cached,
            'free': free,
            'total': total
        }
    return None


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def get_total_episodes_from_logs():
    """Get total episodes completed from all log files."""
    log_file = Path("training_100k_restarts/logs/monitor.csv")
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            return len(lines) - 2  # Subtract headers
        except:
            return 0
    return 0


class ChunkTrainingCallback(BaseCallback):
    """Training callback for chunk-based training with process isolation."""
    
    def __init__(self, 
                 save_path: str,
                 start_episode: int,
                 chunk_episodes: int,
                 verbose: int = 1):
        super().__init__(verbose)
        
        self.save_path = Path(save_path)
        self.model_dir = self.save_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_episode = start_episode
        self.chunk_episodes = chunk_episodes
        self.target_episode = start_episode + chunk_episodes
        
        self.start_time = None
        self.episodes_in_chunk = 0
        self.episode_rewards = []
        self.best_reward = -float('inf')
        
    def _init_callback(self) -> None:
        """Initialize chunk training."""
        self.start_time = time.time()
        
        print(f"🚀 GPU Training Chunk: Episodes {self.start_episode} → {self.target_episode}")
        print(f"=" * 60)
        print(f"📊 Chunk size: {self.chunk_episodes} episodes")
        print(f"🖥️  Device: {self.model.device}")
        
        # Show GPU memory info
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"🧠 GPU Memory: {gpu_info['allocated']:.1f}GB allocated, {gpu_info['free']:.1f}GB free")
        print()
        
    def _on_step(self) -> bool:
        """Track progress within chunk."""
        
        # Update episode count within this chunk
        if hasattr(self.training_env, 'get_attr'):
            try:
                current_episodes = 0
                for env in self.training_env.get_attr('monitor'):
                    if hasattr(env, 'episode_rewards'):
                        current_episodes = len(env.episode_rewards)
                        self.episode_rewards = env.episode_rewards
                
                self.episodes_in_chunk = current_episodes
                
            except:
                current_session_steps = self.n_calls
                self.episodes_in_chunk = current_session_steps // 400
        
        # Progress update every 10 episodes
        if self.episodes_in_chunk % 10 == 0 and self.episodes_in_chunk > 0:
            self._print_progress()
        
        # Memory cleanup every 25 episodes
        if self.episodes_in_chunk % 25 == 0:
            clear_gpu_memory()
        
        # Check chunk completion
        if self.episodes_in_chunk >= self.chunk_episodes:
            print(f"\\n✅ Chunk Complete: {self.chunk_episodes} episodes")
            self._save_final_model()
            return False
        
        return True
        
    def _print_progress(self):
        """Print chunk progress."""
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        if recent_rewards:
            avg_recent_reward = np.mean(recent_rewards)
            max_recent_reward = max(recent_rewards)
            self.best_reward = max(self.best_reward, max_recent_reward)
        else:
            avg_recent_reward = 0
            max_recent_reward = 0
        
        episodes_per_minute = self.episodes_in_chunk / elapsed_minutes if elapsed_minutes > 0 else 0
        progress_percent = (self.episodes_in_chunk / self.chunk_episodes) * 100
        
        # GPU memory info
        gpu_info = get_gpu_memory_info()
        gpu_str = f" (GPU: {gpu_info['allocated']:.1f}GB)" if gpu_info else ""
        
        global_episode = self.start_episode + self.episodes_in_chunk
        
        print(f"\\n📊 Episode {global_episode} (Chunk: {self.episodes_in_chunk}/{self.chunk_episodes}, {progress_percent:.1f}%)")
        print(f"   📈 Recent: {avg_recent_reward:.2f} avg, {max_recent_reward:.2f} max (Best: {self.best_reward:.2f})")
        print(f"   ⚡ Speed: {episodes_per_minute:.1f} episodes/min{gpu_str}")
    
    def _save_final_model(self):
        """Save model at end of chunk."""
        final_episode = self.start_episode + self.episodes_in_chunk
        
        # Save chunk completion model
        chunk_path = self.model_dir / f"chunk_{final_episode:06d}.zip"
        self.model.save(str(chunk_path))
        
        # Always update latest model
        latest_path = self.model_dir / "latest_model.zip"
        self.model.save(str(latest_path))
        
        print(f"💾 Chunk saved: {chunk_path.name}")


def main():
    """Run chunk-based training with process isolation."""
    
    parser = argparse.ArgumentParser(description='GPU Training with Process Isolation')
    parser.add_argument('--start_episode', type=int, default=0,
                       help='Episode number to start from')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of episodes in this chunk')
    parser.add_argument('--total_target', type=int, default=5000,
                       help='Total target episodes for full training')
    
    args = parser.parse_args()
    
    print(f"🚀 Genesis GPU Training - Process Isolation")
    print(f"=" * 50)
    print(f"📊 Chunk: Episodes {args.start_episode} → {args.start_episode + args.num_episodes}")
    print(f"🎯 Total target: {args.total_target:,} episodes")
    print(f"🛡️  Memory protection: Process isolation active")
    print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  No GPU detected, will use CPU")
    
    # Setup directories
    training_dir = Path("./training_5k_restarts")
    log_dir = training_dir / "logs"
    model_dir = training_dir / "models"
    
    training_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    def make_env():
        env = QuadrupedWalkingEnv(
            episode_length=400,
            render_mode=None,
            simulation_fps=60,
            control_freq=15,
            target_velocity=0.6
        )
        return Monitor(env, str(log_dir), allow_early_resets=True)
    
    env = DummyVecEnv([make_env])
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create or load model
    latest_model_path = model_dir / "latest_model.zip"
    
    if args.start_episode > 0 and latest_model_path.exists():
        print(f"🔄 Loading model from episode {args.start_episode}")
        model = PPO.load(str(latest_model_path), env=env, device=device)
        print(f"✅ Model loaded successfully")
    else:
        print(f"🆕 Creating new model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=32,  # Memory-optimized
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={
                "net_arch": [256, 256]  # Memory-optimized
            },
            verbose=1,
            device=device
        )
    
    print(f"🧠 Model device: {model.device}")
    print()
    
    # Create chunk callback
    chunk_callback = ChunkTrainingCallback(
        save_path=str(training_dir),
        start_episode=args.start_episode,
        chunk_episodes=args.num_episodes
    )
    
    print(f"🚀 Starting chunk training...")
    print(f"   ⚡ GPU acceleration with memory isolation")
    print(f"   📊 Progress updates every 10 episodes")
    print(f"   🧹 Memory cleanup every 25 episodes")
    print()
    
    start_time = datetime.now()
    
    try:
        # Train for the chunk
        total_steps = args.num_episodes * 400
        
        model.learn(
            total_timesteps=total_steps,
            callback=[chunk_callback],
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        print(f"\\n🏆 Chunk completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n⏸️  Chunk interrupted by user")
        
        # Save current progress
        interrupt_path = model_dir / f"interrupt_{args.start_episode + chunk_callback.episodes_in_chunk:06d}.zip"
        model.save(str(interrupt_path))
        print(f"   💾 Progress saved: {interrupt_path}")
        
    except Exception as e:
        print(f"\\n❌ Chunk error: {e}")
        
        # Emergency save
        try:
            emergency_path = model_dir / f"emergency_{args.start_episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            model.save(str(emergency_path))
            print(f"   💾 Emergency save: {emergency_path}")
        except:
            print("   ❌ Could not save emergency checkpoint")
            
    finally:
        env.close()
        clear_gpu_memory()
        
        # Final statistics
        end_time = datetime.now()
        duration = end_time - start_time
        final_episodes = chunk_callback.episodes_in_chunk
        global_episodes = args.start_episode + final_episodes
        
        print(f"\\n📊 Chunk Summary:")
        print(f"   ⏱️  Duration: {duration}")
        print(f"   📈 Episodes completed: {final_episodes}/{args.num_episodes}")
        print(f"   📊 Global progress: {global_episodes:,}/{args.total_target:,} ({global_episodes/args.total_target*100:.1f}%)")
        
        # Next chunk info
        if global_episodes < args.total_target:
            next_start = global_episodes
            print(f"\\n🔄 Next chunk command:")
            print(f"   python train_with_restarts.py --start_episode {next_start} --num_episodes {args.num_episodes}")


if __name__ == "__main__":
    main()
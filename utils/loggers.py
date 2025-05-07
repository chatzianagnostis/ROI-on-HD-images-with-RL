from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import json
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger(BaseCallback):
    """
    A logger that clearly separates each episode with visual delimiters,
    shows detailed per-action rewards, saves metrics to txt files,
    and logs to TensorBoard. Each training run gets its own folder.
    """
    def __init__(self, log_dir="logs", verbose=1):
        super(TrainingLogger, self).__init__(verbose)
        
        # Create a timestamped run directory for this specific training session
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(log_dir, "runs", f"run_{timestamp}")
        self.ppo_dir = os.path.join(self.run_dir, "ppo_")
        self.txt_dir = os.path.join(self.run_dir, "txt_logs")
        
        # Create all necessary directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "runs"), exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ppo_dir, exist_ok=True)
        os.makedirs(self.txt_dir, exist_ok=True)
        
        # Initialize TensorBoard writer for this specific run
        self.tb_writer = SummaryWriter(log_dir=self.run_dir)
        
        # Track current episode
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_counter = 0
        self.step_counter = 0
        
        # Track action stats
        self.action_counts = defaultdict(int)
        self.action_rewards = defaultdict(list)
        
        # Action names for clarity
        self.action_names = ["Move Up", "Move Down", "Move Left", "Move Right", 
                           "Place Bbox", "Remove Bbox", "End Episode"]
        
        # Create log file for current session
        self.session_log_path = os.path.join(self.txt_dir, "training_session.txt")
        
        # Print initial episode header and log run information
        self._log_run_info()
        self._print_episode_header()
    
    def _log_run_info(self):
        """Log information about this training run"""
        run_info = f"ROI DETECTION AGENT - TRAINING RUN\n"
        run_info += f"Run directory: {self.run_dir}\n"
        run_info += f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        run_info += f"{'='*60}\n\n"
        
        # Save to run directory
        run_info_path = os.path.join(self.run_dir, "run_info.txt")
        with open(run_info_path, 'w') as f:
            f.write(run_info)
        
        # Also print to console
        print(run_info)
        
        # Save to session log
        with open(self.session_log_path, 'w') as f:
            f.write(run_info)
        
    def _on_step(self) -> bool:
        # Get locals from training
        if "actions" in self.locals and "rewards" in self.locals:
            action = int(self.locals["actions"][0])
            reward = float(self.locals["rewards"][0])
            
            # Track for current episode
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            
            # Track overall
            self.action_counts[action] += 1
            self.action_rewards[action].append(reward)
            
            # Increment step counter
            self.step_counter += 1
            
            # Print action and reward in real-time
            action_name = self.action_names[action] if action < len(self.action_names) else f"Unknown({action})"
            log_text = f"Action: {action_name}\nReward: {reward:.4f}\n"
            print(log_text)
            
            # Write to log file
            with open(self.session_log_path, 'a') as f:
                f.write(log_text)
        
        # Handle episode end
        if self.locals.get("dones")[0]:
            self.episode_counter += 1
            
            # Get episode info
            ep_info = self.locals.get("infos")[0].get("episode")
            if ep_info:
                total_reward = ep_info["r"]
                
                # Print episode footer with summary
                summary_text = self._print_episode_footer(total_reward, self.locals.get("infos")[0])
                
                # Write summary to log file
                with open(self.session_log_path, 'a') as f:
                    f.write(summary_text)
                
                # Save episode data
                self._save_episode_data(total_reward, self.locals.get("infos")[0])
                
                # Log to TensorBoard
                self._log_to_tensorboard(total_reward, self.locals.get("infos")[0])
            
            # Reset episode tracking
            self.episode_actions = []
            self.episode_rewards = []
            self.step_counter = 0
            
            # Print header for next episode
            header_text = self._print_episode_header()
            with open(self.session_log_path, 'a') as f:
                f.write(header_text)
        
        return True
    
    def _print_episode_header(self):
        """Print a clear header for the start of an episode"""
        header_text = f"\n{'='*60}\nEPISODE {self.episode_counter + 1} START\n{'='*60}\n\n"
        print(header_text)
        return header_text
    
    def _print_episode_footer(self, total_reward, info):
        """Print a footer with episode summary"""
        footer_text = f"\n{'-'*60}\nEPISODE {self.episode_counter + 1} COMPLETE\n{'-'*60}\n"
        footer_text += f"Total reward: {total_reward:.4f}\n"
        footer_text += f"Steps taken: {self.step_counter}\n"
        
        # Add metrics if available
        if 'metrics' in info:
            metrics = info['metrics']
            footer_text += "\nFinal Metrics:\n"
            
            # Coverage score
            if 'coverage_score' in metrics:
                footer_text += f"  Coverage score: {metrics['coverage_score']:.4f}\n"
            
            # ROI matching score
            if 'roi_matching_score' in metrics:
                footer_text += f"  ROI matching score: {metrics['roi_matching_score']:.4f}\n"
            
            # Efficiency score
            if 'efficiency_score' in metrics:
                footer_text += f"  Efficiency score: {metrics['efficiency_score']:.4f}\n"
            
            # Overlap penalty
            if 'overlap_penalty' in metrics:
                footer_text += f"  Overlap penalty: {metrics['overlap_penalty']:.4f}\n"
            
            # ROI counts
            if 'optimal_count' in metrics and 'placed_count' in metrics:
                footer_text += f"  Optimal ROI count: {metrics['optimal_count']}\n"
                footer_text += f"  Placed ROI count: {metrics['placed_count']}\n"
            
            # Time info
            if 'time' in metrics:
                time_info = metrics['time']
                footer_text += f"  Time used: {time_info['elapsed']:.2f}s / {time_info['limit']:.2f}s\n"
        
        # Calculate action distribution
        action_counts = {}
        for action in self.episode_actions:
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        # Add action distribution
        footer_text += "\nAction Distribution:\n"
        for action in range(len(self.action_names)):
            count = action_counts.get(action, 0)
            if self.episode_actions:
                percentage = (count / len(self.episode_actions)) * 100
            else:
                percentage = 0
            footer_text += f"  {self.action_names[action]}: {count} ({percentage:.1f}%)\n"
        
        # Add reward statistics
        footer_text += "\nReward Statistics:\n"
        footer_text += f"  Total Reward: {total_reward:.4f}\n"
        if self.episode_rewards:
            footer_text += f"  Average Reward: {np.mean(self.episode_rewards):.4f}\n"
            footer_text += f"  Min Reward: {min(self.episode_rewards):.4f}\n"
            footer_text += f"  Max Reward: {max(self.episode_rewards):.4f}\n"
        
        # Add average reward by action
        footer_text += "\nAverage Reward by Action:\n"
        for action in range(len(self.action_names)):
            rewards = [r for a, r in zip(self.episode_actions, self.episode_rewards) if a == action]
            if rewards:
                footer_text += f"  {self.action_names[action]}: {np.mean(rewards):.4f}\n"
            else:
                footer_text += f"  {self.action_names[action]}: N/A\n"
        
        print(footer_text)
        return footer_text
    
    def _save_episode_data(self, total_reward, info):
        """Save episode data to file"""
        episode_data = {
            "episode": self.episode_counter,
            "timestep": self.num_timesteps,
            "total_reward": total_reward,
            "actions": self.episode_actions.copy(),
            "rewards": self.episode_rewards.copy(),
            "step_count": self.step_counter
        }
        
        # Add metrics if available
        if 'metrics' in info:
            episode_data['metrics'] = info['metrics']
        
        # Save to json file in the ppo_ directory
        filepath = os.path.join(self.ppo_dir, f"episode_{self.episode_counter}.json")
        with open(filepath, 'w') as f:
            json.dump(episode_data, f)
        
        # Also save a separate metrics-only text file for this episode
        metrics_filepath = os.path.join(self.txt_dir, f"metrics_ep_{self.episode_counter}.txt")
        with open(metrics_filepath, 'w') as f:
            f.write(f"Episode {self.episode_counter} Metrics\n")
            f.write(f"Total Reward: {total_reward:.4f}\n")
            f.write(f"Steps Taken: {self.step_counter}\n\n")
            
            if 'metrics' in info:
                metrics = info['metrics']
                metrics_line = "Final Metrics: "
                
                if 'coverage_score' in metrics:
                    metrics_line += f"Coverage score: {metrics['coverage_score']:.4f} "
                
                if 'roi_matching_score' in metrics:
                    metrics_line += f"ROI matching score: {metrics['roi_matching_score']:.4f} "
                
                if 'efficiency_score' in metrics:
                    metrics_line += f"Efficiency score: {metrics['efficiency_score']:.4f} "
                
                if 'overlap_penalty' in metrics:
                    metrics_line += f"Overlap penalty: {metrics['overlap_penalty']:.4f} "
                
                if 'optimal_count' in metrics and 'placed_count' in metrics:
                    metrics_line += f"Optimal ROI count: {metrics['optimal_count']} "
                    metrics_line += f"Placed ROI count: {metrics['placed_count']} "
                
                if 'time' in metrics:
                    time_info = metrics['time']
                    metrics_line += f"Time used: {time_info['elapsed']:.2f}s / {time_info['limit']:.2f}s"
                
                f.write(metrics_line + "\n")

    def _log_to_tensorboard(self, total_reward, info):
        """Log metrics to TensorBoard with standard PPO naming conventions"""
        # Log episode reward
        self.tb_writer.add_scalar("rollout/ep_rew_mean", total_reward, self.num_timesteps)
        self.tb_writer.add_scalar("rollout/ep_len_mean", self.step_counter, self.num_timesteps)
        
        # Log action distribution with standard PPO format
        action_counts = {}
        for action in self.episode_actions:
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        for action in range(len(self.action_names)):
            count = action_counts.get(action, 0)
            if self.episode_actions:
                percentage = (count / len(self.episode_actions)) * 100
            else:
                percentage = 0
            
            # Use standard PPO naming convention
            self.tb_writer.add_scalar(f"PPO/{self.action_names[action]}_frequency", percentage, self.num_timesteps)
        
        # Log metrics if available with PPO prefix
        if 'metrics' in info:
            metrics = info['metrics']
            
            # Coverage score
            if 'coverage_score' in metrics:
                self.tb_writer.add_scalar("PPO/coverage_score", metrics['coverage_score'], self.num_timesteps)
            
            # ROI matching score
            if 'roi_matching_score' in metrics:
                self.tb_writer.add_scalar("PPO/roi_matching_score", metrics['roi_matching_score'], self.num_timesteps)
            
            # Efficiency score
            if 'efficiency_score' in metrics:
                self.tb_writer.add_scalar("PPO/efficiency_score", metrics['efficiency_score'], self.num_timesteps)
            
            # Overlap penalty
            if 'overlap_penalty' in metrics:
                self.tb_writer.add_scalar("PPO/overlap_penalty", metrics['overlap_penalty'], self.num_timesteps)
            
            # ROI counts
            if 'optimal_count' in metrics and 'placed_count' in metrics:
                self.tb_writer.add_scalar("PPO/optimal_roi_count", metrics['optimal_count'], self.num_timesteps)
                self.tb_writer.add_scalar("PPO/placed_roi_count", metrics['placed_count'], self.num_timesteps)
                self.tb_writer.add_scalar("PPO/roi_count_difference", 
                                        metrics['placed_count'] - metrics['optimal_count'], 
                                        self.num_timesteps)
    
    def on_training_end(self):
        """Called when training ends"""
        # Print final statistics
        final_stats = f"\n{'#'*80}\nTRAINING COMPLETE - FINAL STATISTICS\n{'#'*80}\n"
        final_stats += f"Total Episodes: {self.episode_counter}\n"
        final_stats += f"Total Timesteps: {self.num_timesteps}\n"
        final_stats += f"Run directory: {self.run_dir}\n"
        
        # Action distribution
        total_actions = sum(self.action_counts.values())
        
        final_stats += "\nOverall Action Distribution:\n"
        for action in range(len(self.action_names)):
            count = self.action_counts.get(action, 0)
            if total_actions > 0:
                percentage = (count / total_actions) * 100
            else:
                percentage = 0
            final_stats += f"  {self.action_names[action]}: {count} ({percentage:.1f}%)\n"
        
        # Overall reward statistics by action
        final_stats += "\nOverall Average Reward by Action:\n"
        for action in range(len(self.action_names)):
            rewards = self.action_rewards.get(action, [])
            if rewards:
                avg_reward = np.mean(rewards)
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                final_stats += f"  {self.action_names[action]}: {avg_reward:.4f}\n"
                final_stats += f"    Min: {min_reward:.4f}, Max: {max_reward:.4f}, Count: {len(rewards)}\n"
            else:
                final_stats += f"  {self.action_names[action]}: N/A\n"
        
        print(final_stats)
        
        # Save final stats to file
        final_stats_path = os.path.join(self.txt_dir, "final_training_stats.txt")
        with open(final_stats_path, 'w') as f:
            f.write(final_stats)
        
        # Also save a copy of the final stats to the ppo_ directory for convenience
        ppo_stats_path = os.path.join(self.ppo_dir, "final_training_stats.json")
        with open(ppo_stats_path, 'w') as f:
            stats_dict = {
                "episodes": self.episode_counter,
                "timesteps": self.num_timesteps,
                "run_dir": self.run_dir,
                "action_counts": dict(self.action_counts),
                "action_rewards": {action: {
                    "mean": float(np.mean(rewards)) if rewards else None,
                    "min": float(np.min(rewards)) if rewards else None,
                    "max": float(np.max(rewards)) if rewards else None,
                    "count": len(rewards)
                } for action, rewards in self.action_rewards.items()}
            }
            json.dump(stats_dict, f, indent=2)
        
        # Save a run_complete marker file
        with open(os.path.join(self.run_dir, "run_complete.txt"), 'w') as f:
            f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total episodes: {self.episode_counter}\n")
            f.write(f"Total timesteps: {self.num_timesteps}\n")
        
        # Close TensorBoard writer
        self.tb_writer.close()
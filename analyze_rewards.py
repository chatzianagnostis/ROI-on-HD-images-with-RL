import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_action_rewards(log_dir="logs", run_id=None):
    """
    Analyze and visualize rewards per action from training logs.
    
    Args:
        log_dir: Base logs directory
        run_id: Specific run ID to analyze (if None, will use the latest run)
    """
    # Find available run directories
    run_dirs = sorted(glob.glob(os.path.join(log_dir, "runs", "run_*")))
    
    if not run_dirs:
        print(f"No run directories found in {os.path.join(log_dir, 'runs')}.")
        return
    
    # Select the run directory to analyze
    if run_id is not None:
        # Find a specific run by partial ID match
        matching_runs = [d for d in run_dirs if run_id in d]
        if not matching_runs:
            print(f"No runs matching ID '{run_id}' found.")
            print("Available runs:")
            for run_dir in run_dirs:
                print(f"  {os.path.basename(run_dir)}")
            return
        run_dir = matching_runs[0]
    else:
        # Use the most recent run by default
        run_dir = run_dirs[-1]
    
    print(f"Analyzing run: {os.path.basename(run_dir)}")
    
    # Find episode files in the ppo_ directory
    ppo_dir = os.path.join(run_dir, "ppo_")
    episode_files = sorted(glob.glob(os.path.join(ppo_dir, "episode_*.json")))
    
    if not episode_files:
        print(f"No episode data found in {ppo_dir}")
        return
    
    # Action names
    action_names = ["Move Up", "Move Down", "Move Left", "Move Right", 
                   "Place Bbox", "Remove Bbox", "End Episode"]
    
    # Collect action rewards
    action_rewards = defaultdict(list)
    action_counts = defaultdict(int)
    
    for ep_file in episode_files:
        with open(ep_file, 'r') as f:
            episode = json.load(f)
            
            for action, reward in zip(episode.get("actions", []), episode.get("rewards", [])):
                action_rewards[action].append(reward)
                action_counts[action] += 1
    
    # Create output directory within the run folder
    output_dir = os.path.join(run_dir, "reward_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate reward statistics
    print("\n==== ACTION REWARD STATISTICS ====")
    print(f"{'Action':<15} {'Count':<8} {'Avg Reward':<12} {'Min':<8} {'Max':<8}")
    print("-" * 60)
    
    # Prepare data for plotting
    labels = []
    values = []
    errors = []
    counts = []
    
    for action in range(len(action_names)):
        rewards = action_rewards.get(action, [])
        count = action_counts.get(action, 0)
        
        if count > 0:
            avg_reward = np.mean(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            std_reward = np.std(rewards)
            
            print(f"{action_names[action]:<15} {count:<8} {avg_reward:<12.6f} {min_reward:<8.4f} {max_reward:<8.4f}")
            
            labels.append(action_names[action])
            values.append(avg_reward)
            errors.append(std_reward)
            counts.append(count)
    
    # Create bar chart of average rewards
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, yerr=errors, capsize=10)
    plt.title('Average Reward per Action')
    plt.xlabel('Action')
    plt.ylabel('Average Reward')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    reward_plot_path = os.path.join(output_dir, "action_rewards.png")
    plt.savefig(reward_plot_path)
    plt.close()
    
    # Create bar chart of action counts
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.title('Action Usage Counts')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    count_plot_path = os.path.join(output_dir, "action_counts.png")
    plt.savefig(count_plot_path)
    plt.close()
    
    # Create reward distribution plot
    plt.figure(figsize=(12, 8))
    
    for action in range(len(action_names)):
        rewards = action_rewards.get(action, [])
        if len(rewards) > 10:  # Only plot if we have enough data
            plt.hist(rewards, bins=30, alpha=0.5, label=action_names[action])
    
    plt.title('Reward Distribution by Action')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    dist_plot_path = os.path.join(output_dir, "reward_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    
    print(f"\nPlots saved to {output_dir} directory")
    print(f"  - {reward_plot_path}")
    print(f"  - {count_plot_path}")
    print(f"  - {dist_plot_path}")
    
    # Track reward trends over episodes
    try:
        # Prepare episode numbers and reward data
        episode_nums = []
        episode_rewards = []
        action_trends = {action: [] for action in range(len(action_names))}
        
        for i, ep_file in enumerate(episode_files):
            with open(ep_file, 'r') as f:
                ep_data = json.load(f)
                
                episode_nums.append(ep_data.get("episode", i))
                episode_rewards.append(ep_data.get("total_reward", 0))
                
                # Collect rewards per action for this episode
                ep_action_rewards = defaultdict(list)
                
                for action, reward in zip(ep_data.get("actions", []), ep_data.get("rewards", [])):
                    ep_action_rewards[action].append(reward)
                
                # Calculate average reward per action for this episode
                for action in range(len(action_names)):
                    if action in ep_action_rewards and ep_action_rewards[action]:
                        action_trends[action].append(np.mean(ep_action_rewards[action]))
                    else:
                        action_trends[action].append(None)  # No data for this action
        
        # Plot overall episode reward trend
        plt.figure(figsize=(12, 6))
        plt.plot(episode_nums, episode_rewards, marker='o', linestyle='-', color='blue')
        plt.title('Episode Total Reward Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.tight_layout()
        
        episode_trend_path = os.path.join(output_dir, "episode_reward_trend.png")
        plt.savefig(episode_trend_path)
        plt.close()
        
        # Plot reward trends per action
        plt.figure(figsize=(12, 6))
        
        for action in range(len(action_names)):
            # Filter out None values
            valid_indices = [i for i, reward in enumerate(action_trends[action]) if reward is not None]
            valid_episodes = [episode_nums[i] for i in valid_indices]
            valid_rewards = [action_trends[action][i] for i in valid_indices]
            
            if valid_rewards:  # Only plot if we have data
                plt.plot(valid_episodes, valid_rewards, marker='o', linestyle='-', label=action_names[action])
        
        plt.title('Average Reward Trends by Action')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        action_trend_path = os.path.join(output_dir, "action_reward_trends.png")
        plt.savefig(action_trend_path)
        plt.close()
        
        print(f"  - {episode_trend_path}")
        print(f"  - {action_trend_path}")
    except Exception as e:
        print(f"Could not generate trend plots: {e}")
    
    # Create detailed report
    report_path = os.path.join(output_dir, "action_reward_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"ROI DETECTION AGENT - ACTION REWARD ANALYSIS\n")
        f.write(f"Run: {os.path.basename(run_dir)}\n")
        f.write("=========================================\n\n")
        
        # Write summary stats
        for action in range(len(action_names)):
            rewards = action_rewards.get(action, [])
            count = action_counts.get(action, 0)
            
            if count > 0:
                avg_reward = np.mean(rewards)
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                std_reward = np.std(rewards)
                
                f.write(f"{action_names[action]}:\n")
                f.write(f"  Count: {count}\n")
                f.write(f"  Average Reward: {avg_reward:.6f}\n")
                f.write(f"  Min/Max Reward: {min_reward:.6f} / {max_reward:.6f}\n")
                f.write(f"  Standard Deviation: {std_reward:.6f}\n\n")
                
                # Calculate percentage of positive/negative rewards
                positive_count = sum(1 for r in rewards if r > 0)
                zero_count = sum(1 for r in rewards if r == 0)
                negative_count = sum(1 for r in rewards if r < 0)
                
                f.write(f"  Reward Distribution:\n")
                f.write(f"    Positive: {positive_count} ({positive_count/count*100:.1f}%)\n")
                f.write(f"    Zero: {zero_count} ({zero_count/count*100:.1f}%)\n")
                f.write(f"    Negative: {negative_count} ({negative_count/count*100:.1f}%)\n\n")
        
        # Add notes about final rewards
        f.write("\nNOTES ON REWARDS:\n")
        f.write("- Movement actions (Up, Down, Left, Right) typically receive small shaping rewards\n")
        f.write("- Place Bbox can receive negative rewards for overlapping with existing boxes\n")
        f.write("- The End Episode action triggers the final reward calculation which includes:\n")
        f.write("  * Coverage score (whether all annotations are covered by ROIs)\n")
        f.write("  * ROI matching score (how well placed ROIs match optimal ones)\n")
        f.write("  * Efficiency score (using close to optimal number of ROIs)\n")
        f.write("  * Overlap penalty (for excessive overlap between ROIs)\n")
    
    print(f"  - {report_path}")
    
    # Save a JSON version of the analysis
    json_report_path = os.path.join(output_dir, "reward_analysis.json")
    json_report = {
        "run": os.path.basename(run_dir),
        "episodes_analyzed": len(episode_files),
        "actions": {}
    }
    
    for action in range(len(action_names)):
        rewards = action_rewards.get(action, [])
        count = action_counts.get(action, 0)
        
        if count > 0:
            json_report["actions"][action_names[action]] = {
                "count": count,
                "avg_reward": float(np.mean(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "std_reward": float(np.std(rewards)),
                "positive_pct": float(sum(1 for r in rewards if r > 0) / count * 100),
                "zero_pct": float(sum(1 for r in rewards if r == 0) / count * 100),
                "negative_pct": float(sum(1 for r in rewards if r < 0) / count * 100)
            }
    
    with open(json_report_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"  - {json_report_path}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze action rewards from training logs")
    parser.add_argument("--log_dir", default="logs", help="Base log directory")
    parser.add_argument("--run_id", default=None, help="Specific run ID to analyze (default: latest run)")
    
    args = parser.parse_args()
    analyze_action_rewards(args.log_dir, args.run_id)
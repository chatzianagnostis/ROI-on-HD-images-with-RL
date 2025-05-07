import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_action_rewards(log_dir="logs"):
    """
    Analyze and visualize rewards per action from training logs
    """
    # Load epoch files
    epoch_files = sorted(glob.glob(os.path.join(log_dir, "episode_*.json")))
    
    if not epoch_files:
        print("No epoch data found. Run training first.")
        return
    
    # Load episode files to get per-step rewards
    episode_files = sorted(glob.glob(os.path.join(log_dir, "episode_*.json")))
    
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
    
    # Create output directory
    output_dir = "reward_analysis"
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
        epoch_trends = {action: [] for action in range(len(action_names))}
        epoch_nums = []
        
        for epoch_file in epoch_files:
            with open(epoch_file, 'r') as f:
                epoch_data = json.load(f)
                epoch_nums.append(epoch_data.get("epoch", 0))
                
                # Get rewards per action
                action_stats = epoch_data.get("action_reward_stats", {})
                
                for action in range(len(action_names)):
                    action_name = action_names[action]
                    if action_name in action_stats and "avg_reward" in action_stats[action_name]:
                        epoch_trends[action].append(action_stats[action_name]["avg_reward"])
                    else:
                        # Use None for epochs without this action
                        epoch_trends[action].append(None)
        
        # Plot reward trends
        plt.figure(figsize=(12, 6))
        
        for action in range(len(action_names)):
            # Filter out None values
            valid_indices = [i for i, reward in enumerate(epoch_trends[action]) if reward is not None]
            valid_epochs = [epoch_nums[i] for i in valid_indices]
            valid_rewards = [epoch_trends[action][i] for i in valid_indices]
            
            if valid_rewards:  # Only plot if we have data
                plt.plot(valid_epochs, valid_rewards, marker='o', linestyle='-', label=action_names[action])
        
        plt.title('Average Reward Trends by Action')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        trend_plot_path = os.path.join(output_dir, "reward_trends.png")
        plt.savefig(trend_plot_path)
        plt.close()
        
        print(f"  - {trend_plot_path}")
    except Exception as e:
        print(f"Could not generate trend plot: {e}")
    
    # Create detailed report
    report_path = os.path.join(output_dir, "action_reward_report.txt")
    with open(report_path, 'w') as f:
        f.write("ROI DETECTION AGENT - ACTION REWARD ANALYSIS\n")
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
    return output_dir

if __name__ == "__main__":
    analyze_action_rewards()
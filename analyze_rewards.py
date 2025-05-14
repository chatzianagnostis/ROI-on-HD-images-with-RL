import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

# --- Configuration & Constants ---
ACTION_NAMES = [
    "Move Up", "Move Down", "Move Left", "Move Right",
    "Place Bbox", "Remove Bbox", "End Episode"
]
# Colors for consistent plotting of actions
ACTION_COLORS = plt.cm.get_cmap('tab10', len(ACTION_NAMES))

# --- Helper Functions ---
def _moving_average(data, window_size):
    if not data or window_size <= 0:
        return []
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def _ensure_output_dir(base_dir, dirname="reward_policy_analysis"):
    output_dir = os.path.join(base_dir, dirname)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def _save_plot(fig, output_dir, filename, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  - Saved plot: {filename}")

# --- Data Loading and Processing ---
def load_episode_data(ppo_dir):
    """Loads all episode JSON files from the ppo_ directory."""
    episode_files = sorted(glob.glob(os.path.join(ppo_dir, "episode_*.json")))
    if not episode_files:
        print(f"No episode data found in {ppo_dir}")
        return []

    all_episode_data = []
    for i, ep_file in enumerate(episode_files):
        try:
            with open(ep_file, 'r') as f:
                data = json.load(f)
                # Ensure basic structure and add episode number if missing
                if "episode_number" not in data: # Use custom key
                    data["episode_number"] = data.get("episode", i) # Fallback to "episode" or file order
                if "actions" not in data: data["actions"] = []
                if "rewards" not in data: data["rewards"] = []
                if "total_reward" not in data: data["total_reward"] = sum(data["rewards"])
                all_episode_data.append(data)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {ep_file}. Skipping.")
        except Exception as e:
            print(f"Warning: Error processing {ep_file}: {e}. Skipping.")
    
    # Sort by episode number to ensure correct temporal order
    all_episode_data.sort(key=lambda x: x["episode_number"])
    return all_episode_data

# --- Analysis Functions ---

def analyze_overall_performance(all_episode_data, output_dir, window_size=50):
    """Analyzes and plots total episode rewards over time."""
    print("\n--- Analyzing Overall Performance ---")
    if not all_episode_data:
        print("No episode data to analyze for overall performance.")
        return

    episode_numbers = [ep["episode_number"] for ep in all_episode_data]
    total_rewards = [ep["total_reward"] for ep in all_episode_data]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(episode_numbers, total_rewards, label='Total Reward per Episode', alpha=0.6, color='blue')
    
    if len(total_rewards) >= window_size:
        smoothed_rewards = _moving_average(total_rewards, window_size)
        # Adjust episode numbers for smoothed plot to align centers of windows
        smoothed_ep_numbers = episode_numbers[window_size-1:] 
        if len(smoothed_ep_numbers) == len(smoothed_rewards): # Check for length consistency
             ax.plot(smoothed_ep_numbers, smoothed_rewards, label=f'{window_size}-Episode Moving Avg', color='red', linewidth=2)
        else:
            print(f"Warning: Length mismatch for smoothing overall performance. Smoothed_ep: {len(smoothed_ep_numbers)}, Smoothed_rewards: {len(smoothed_rewards)}")


    ax.set_title('Overall Agent Performance: Total Reward Over Episodes')
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    _save_plot(fig, output_dir, "1_overall_episode_reward_trend.png")

def analyze_action_distributions_and_policy(all_episode_data, output_dir, num_bins=10):
    """Analyzes action counts and their evolution over time (policy inference)."""
    print("\n--- Analyzing Action Distributions & Policy Evolution ---")
    if not all_episode_data:
        print("No episode data for action distribution analysis.")
        return

    # 1. Aggregate Action Counts and Rewards
    action_counts_total = defaultdict(int)
    action_rewards_total = defaultdict(list)

    for ep in all_episode_data:
        for action_code, reward_val in zip(ep.get("actions", []), ep.get("rewards", [])):
            action_counts_total[action_code] += 1
            action_rewards_total[action_code].append(reward_val)

    # Plot: Overall Action Usage Counts
    fig_counts, ax_counts = plt.subplots(figsize=(12, 7))
    actions_present_codes = sorted(action_counts_total.keys())
    action_labels_present = [ACTION_NAMES[code] for code in actions_present_codes]
    counts_present = [action_counts_total[code] for code in actions_present_codes]
    
    bars = ax_counts.bar(action_labels_present, counts_present, color=[ACTION_COLORS(i) for i in actions_present_codes])
    ax_counts.set_title('Overall Action Usage Counts (Policy Preference)')
    ax_counts.set_xlabel('Action')
    ax_counts.set_ylabel('Total Times Chosen')
    ax_counts.tick_params(axis='x', rotation=45, ha="right")
    ax_counts.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars: # Add count numbers on bars
        yval = bar.get_height()
        ax_counts.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(counts_present), int(yval), ha='center', va='bottom')
    _save_plot(fig_counts, output_dir, "2a_overall_action_counts.png")

    # 2. Temporal Analysis of Action Selection (Policy Evolution)
    if len(all_episode_data) > num_bins * 5 : # Only if enough data for meaningful bins
        episode_bins = np.array_split(all_episode_data, num_bins)
        bin_labels = [f"Ep. {bin_data[0]['episode_number']}-{bin_data[-1]['episode_number']}" for bin_data in episode_bins if bin_data]

        action_percentages_per_bin = defaultdict(list)

        for action_code in range(len(ACTION_NAMES)):
            for bin_data in episode_bins:
                if not bin_data: continue
                total_actions_in_bin = sum(len(ep.get("actions",[])) for ep in bin_data)
                count_this_action_in_bin = sum(ep.get("actions",[]).count(action_code) for ep in bin_data)
                if total_actions_in_bin > 0:
                    action_percentages_per_bin[action_code].append((count_this_action_in_bin / total_actions_in_bin) * 100)
                else:
                    action_percentages_per_bin[action_code].append(0)
        
        fig_temporal_policy, ax_temporal_policy = plt.subplots(figsize=(15, 8))
        for action_code in range(len(ACTION_NAMES)):
            if any(p > 0 for p in action_percentages_per_bin[action_code]): # Plot if action was ever taken
                 ax_temporal_policy.plot(bin_labels, action_percentages_per_bin[action_code],
                                    label=ACTION_NAMES[action_code], marker='o', color=ACTION_COLORS(action_code))

        ax_temporal_policy.set_title('Policy Evolution: Action Selection Percentage Over Training Bins')
        ax_temporal_policy.set_xlabel('Episode Bins')
        ax_temporal_policy.set_ylabel('Percentage of Actions Taken (%)')
        ax_temporal_policy.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_temporal_policy.grid(True, linestyle='--', alpha=0.7)
        ax_temporal_policy.tick_params(axis='x', rotation=45, ha="right")
        _save_plot(fig_temporal_policy, output_dir, "2b_policy_evolution_action_percentages.png")

def analyze_rewards_per_action(all_episode_data, output_dir, window_size=50):
    """Analyzes reward statistics and trends for each action."""
    print("\n--- Analyzing Rewards Per Action ---")
    if not all_episode_data:
        print("No episode data for reward-per-action analysis.")
        return

    action_rewards_total = defaultdict(list)
    for ep in all_episode_data:
        for action_code, reward_val in zip(ep.get("actions", []), ep.get("rewards", [])):
            action_rewards_total[action_code].append(reward_val)

    # 1. Overall Reward Statistics per Action
    print("\n  Overall Reward Statistics per Action:")
    stats_data = []
    for action_code in range(len(ACTION_NAMES)):
        rewards = action_rewards_total.get(action_code, [])
        count = len(rewards)
        if count > 0:
            mean_r, std_r, min_r, max_r = np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)
            stats_data.append({
                "Action": ACTION_NAMES[action_code], "Count": count,
                "Mean Reward": f"{mean_r:.3f}", "Std Dev": f"{std_r:.3f}",
                "Min Reward": f"{min_r:.3f}", "Max Reward": f"{max_r:.3f}"
            })
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
        stats_df.to_csv(os.path.join(output_dir, "3a_action_reward_summary_stats.csv"), index=False)
        print(f"  - Saved CSV: 3a_action_reward_summary_stats.csv")


    # 2. Reward Distributions per Action (Box Plot for better comparison)
    fig_dist, ax_dist = plt.subplots(figsize=(15, 8))
    plot_data = []
    plot_labels = []
    for action_code in range(len(ACTION_NAMES)):
        rewards = action_rewards_total.get(action_code, [])
        if len(rewards) > 5: # Only plot if there's a reasonable amount of data
            plot_data.append(rewards)
            plot_labels.append(ACTION_NAMES[action_code])
    
    if plot_data:
        ax_dist.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red', linewidth=2))
        ax_dist.set_title('Reward Distribution by Action (Box Plot)')
        ax_dist.set_xlabel('Action')
        ax_dist.set_ylabel('Reward Value')
        ax_dist.tick_params(axis='x', rotation=45, ha="right")
        ax_dist.grid(True, axis='y', linestyle='--', alpha=0.7)
        _save_plot(fig_dist, output_dir, "3b_reward_distributions_boxplot.png")

    # 3. Temporal Reward Trends per Action
    fig_temporal_rewards, ax_temporal_rewards = plt.subplots(figsize=(15, 8))
    episode_numbers_global = [ep["episode_number"] for ep in all_episode_data]

    for action_code in range(len(ACTION_NAMES)):
        # Collect rewards for this action chronologically
        rewards_for_action_chronological = []
        ep_indices_for_action = [] # Store original episode numbers where this action occurred

        current_idx = 0
        for ep_idx, ep in enumerate(all_episode_data):
            action_indices_in_ep = [i for i, ac in enumerate(ep.get("actions",[])) if ac == action_code]
            if action_indices_in_ep:
                # Take average reward for this action if taken multiple times in one episode, or just the rewards
                # For simplicity, let's append all rewards and associate with episode number
                for i in action_indices_in_ep:
                    rewards_for_action_chronological.append(ep.get("rewards",[])[i])
                    ep_indices_for_action.append(ep["episode_number"])
        
        if len(rewards_for_action_chronological) >= window_size:
            smoothed_action_rewards = _moving_average(rewards_for_action_chronological, window_size)
            # Align episode numbers for smoothed data
            aligned_ep_numbers = ep_indices_for_action[window_size-1:]
            # Need to be careful if action is sparse, ep_indices_for_action might not be contiguous
            # For plotting, it's better to have a consistent x-axis (original episode numbers)
            # This requires a bit more work to map smoothed values back to original episode scale or plot differently.
            # Simplified approach: plot smoothed vs. its own 'step' count if ep_indices are tricky.
            # Better: Use a rolling mean on a pandas series indexed by episode number where action occurred.

            # For now, plot smoothed rewards against their occurrence index (simplified)
            if len(aligned_ep_numbers) == len(smoothed_action_rewards):
                ax_temporal_rewards.plot(aligned_ep_numbers, smoothed_action_rewards,
                                     label=f'{ACTION_NAMES[action_code]} (smoothed)',
                                     alpha=0.8, color=ACTION_COLORS(action_code))
            # else: # Can also plot raw points if not enough for smoothing or if preferred
            #    ax_temporal_rewards.scatter(ep_indices_for_action, rewards_for_action_chronological,
            #                                label=f'{ACTION_NAMES[action_code]} (raw)',
            #                                alpha=0.3, color=ACTION_COLORS(action_code), s=10)


    ax_temporal_rewards.set_title(f'Smoothed Reward Trends by Action (Window: {window_size})')
    ax_temporal_rewards.set_xlabel('Episode Number (approximate, based on action occurrence)')
    ax_temporal_rewards.set_ylabel('Average Reward for Action')
    ax_temporal_rewards.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_temporal_rewards.grid(True, linestyle='--', alpha=0.7)
    _save_plot(fig_temporal_rewards, output_dir, "3c_action_reward_trends_smoothed.png")


def analyze_key_actions_details(all_episode_data, output_dir):
    """Deeper analysis of 'Place Bbox' and 'Remove Bbox' actions."""
    print("\n--- Analyzing Key Actions (Place/Remove Bbox) ---")
    if not all_episode_data:
        print("No episode data for key action analysis.")
        return

    try:
        place_action_code = ACTION_NAMES.index("Place Bbox")
        remove_action_code = ACTION_NAMES.index("Remove Bbox")
    except ValueError:
        print("Error: 'Place Bbox' or 'Remove Bbox' not in ACTION_NAMES. Skipping key action analysis.")
        return

    place_rewards = []
    remove_rewards = []
    place_ep_numbers = []
    remove_ep_numbers = []

    for ep in all_episode_data:
        ep_num = ep["episode_number"]
        for i, action_code in enumerate(ep.get("actions", [])):
            reward_val = ep.get("rewards", [])[i]
            if action_code == place_action_code:
                place_rewards.append(reward_val)
                place_ep_numbers.append(ep_num)
            elif action_code == remove_action_code:
                remove_rewards.append(reward_val)
                remove_ep_numbers.append(ep_num)

    # Define reward categories for Place Bbox
    # These thresholds depend on your environment's reward design from ROIDetectionEnv.py
    # Example thresholds (adjust these based on your _place_bbox logic):
    PLACE_REWARD_CATEGORIES = {
        "Highly Positive (>5)": (5, float('inf')),
        "Positive (0 to 5)": (0, 5),
        "Slightly Negative (-2 to 0)": (-2, 0), # e.g. no coverage penalty
        "Highly Negative (<-2)": (float('-inf'), -2) # e.g. overlap + no coverage
    }
    
    if place_rewards:
        place_reward_category_counts = defaultdict(int)
        for r in place_rewards:
            for cat_name, (low, high) in PLACE_REWARD_CATEGORIES.items():
                if low < r <= high: # Careful with inclusive/exclusive for zero
                     if r == 0 and low == 0: # Ensure 0 is in "Positive (0 to 5)" if low is 0
                        if cat_name == "Positive (0 to 5)":
                           place_reward_category_counts[cat_name] += 1
                           break
                     elif low < r <=high :
                        place_reward_category_counts[cat_name] += 1
                        break
        
        fig_place, ax_place = plt.subplots(figsize=(10, 7))
        cat_labels = list(place_reward_category_counts.keys())
        cat_counts = [place_reward_category_counts[k] for k in cat_labels]
        ax_place.bar(cat_labels, cat_counts, color='skyblue')
        ax_place.set_title('Reward Categories for "Place Bbox" Action')
        ax_place.set_xlabel('Reward Category')
        ax_place.set_ylabel('Count')
        ax_place.tick_params(axis='x', rotation=45, ha="right")
        _save_plot(fig_place, output_dir, "4a_place_bbox_reward_categories.png")

    # Similar analysis for "Remove Bbox" could be done if you have clear positive/negative categories
    # e.g., positive for removing bad box, negative for removing good box.
    # For now, just a scatter plot of remove rewards vs episode.
    if remove_rewards:
        fig_remove, ax_remove = plt.subplots(figsize=(12,6))
        ax_remove.scatter(remove_ep_numbers, remove_rewards, alpha=0.5, s=15, label="Remove Bbox Rewards")
        ax_remove.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax_remove.set_title('"Remove Bbox" Rewards Over Episodes')
        ax_remove.set_xlabel('Episode Number')
        ax_remove.set_ylabel('Reward Value')
        ax_remove.legend()
        _save_plot(fig_remove, output_dir, "4b_remove_bbox_rewards_scatter.png")


def generate_text_report(all_episode_data, output_dir, run_name):
    """Generates a summary text report."""
    print("\n--- Generating Text Report ---")
    report_path = os.path.join(output_dir, "5_summary_report.txt")

    action_counts_total = defaultdict(int)
    action_rewards_total = defaultdict(list)
    for ep in all_episode_data:
        for action_code, reward_val in zip(ep.get("actions",[]), ep.get("rewards",[])):
            action_counts_total[action_code] += 1
            action_rewards_total[action_code].append(reward_val)

    with open(report_path, 'w') as f:
        f.write(f"ROI DETECTION AGENT - ENHANCED REWARD & POLICY ANALYSIS\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"Total Episodes Analyzed: {len(all_episode_data)}\n")
        f.write("=========================================================\n\n")

        f.write("1. OVERALL PERFORMANCE:\n")
        if all_episode_data:
            total_rewards = [ep["total_reward"] for ep in all_episode_data]
            f.write(f"  - Average Total Reward: {np.mean(total_rewards):.2f}\n")
            f.write(f"  - Std Dev Total Reward: {np.std(total_rewards):.2f}\n")
            f.write(f"  - Min/Max Total Reward: {np.min(total_rewards):.2f} / {np.max(total_rewards):.2f}\n")
        f.write("  (See plot: 1_overall_episode_reward_trend.png)\n\n")

        f.write("2. ACTION SELECTION (POLICY INSIGHTS):\n")
        f.write("  Overall Action Counts (Policy Preference):\n")
        for action_code in sorted(action_counts_total.keys()):
            f.write(f"    - {ACTION_NAMES[action_code]:<15}: {action_counts_total[action_code]} times\n")
        f.write("  (See plots: 2a_overall_action_counts.png, 2b_policy_evolution_action_percentages.png)\n\n")

        f.write("3. REWARD ANALYSIS PER ACTION:\n")
        for action_code in range(len(ACTION_NAMES)):
            rewards = action_rewards_total.get(action_code, [])
            count = len(rewards)
            if count > 0:
                f.write(f"  Action: {ACTION_NAMES[action_code]}\n")
                f.write(f"    - Count: {count}\n")
                f.write(f"    - Avg Reward: {np.mean(rewards):.3f} (Std: {np.std(rewards):.3f})\n")
                f.write(f"    - Min/Max Reward: {np.min(rewards):.3f} / {np.max(rewards):.3f}\n")
                pos_count = sum(1 for r in rewards if r > 0.001) # Threshold for positive
                neg_count = sum(1 for r in rewards if r < -0.001) # Threshold for negative
                zero_ish_count = count - pos_count - neg_count
                f.write(f"    - Positive Rewards (>0.001): {pos_count} ({pos_count/count*100:.1f}%)\n")
                f.write(f"    - Zero-ish Rewards: {zero_ish_count} ({zero_ish_count/count*100:.1f}%)\n")
                f.write(f"    - Negative Rewards (<-0.001): {neg_count} ({neg_count/count*100:.1f}%)\n\n")
        f.write("  (See plots: 3a_action_reward_summary_stats.csv, 3b_reward_distributions_boxplot.png, 3c_action_reward_trends_smoothed.png)\n\n")

        f.write("4. KEY ACTION DETAILS (Place/Remove Bbox):\n")
        f.write("  (See plots: 4a_place_bbox_reward_categories.png, 4b_remove_bbox_rewards_scatter.png)\n\n")

        f.write("5. NOTES FOR DEEPER POLICY/REWARD UNDERSTANDING:\n")
        f.write("  - Examine the 'Policy Evolution' plot (2b) to see how often the agent chooses each action over time.\n")
        f.write("  - Correlate reward trends per action (3c) with overall performance (1).\n")
        f.write("    Does the agent get better at actions that yield high rewards?\n")
        f.write("  - The 'Place Bbox Reward Categories' (4a) can indicate if the agent is learning to make better placements\n")
        f.write("    (e.g., fewer highly negative rewards due to overlap, more positive rewards from good coverage).\n")
        f.write("  - IF MORE DETAILED LOGS WERE AVAILABLE:\n")
        f.write("    - Logging components of the final reward (e.g., coverage_score, matching_score, efficiency_penalty from ROIDetectionEnv)\n")
        f.write("      in the episode JSONs would allow direct tracking of how well the agent learns the specific objectives.\n")
        f.write("    - Logging key state features at each step could enable analysis of state-action-reward relationships.\n")

    print(f"  - Saved text report: {report_path}")

# --- Main Execution ---
def main_analyzer(log_dir="logs", run_id=None, analysis_subdir_name="reward_policy_analysis_v2"):
    """
    Main function to run all analyses.
    """
    # Find and select the run directory
    base_runs_path = os.path.join(log_dir, "runs") # Assuming SB3-like structure from original script
    if not os.path.exists(base_runs_path):
        print(f"Error: Base runs directory not found: {base_runs_path}")
        print(f"Please ensure your logs are structured like: {log_dir}/runs/run_ID_TIMESTAMP/ppo_/episode_*.json")
        return

    run_dirs = sorted([d for d in os.listdir(base_runs_path) if os.path.isdir(os.path.join(base_runs_path, d)) and d.startswith("run_")])

    if not run_dirs:
        print(f"No run directories found in {base_runs_path}.")
        return

    selected_run_name = None
    if run_id is not None:
        matching_runs = [d for d in run_dirs if run_id in d]
        if not matching_runs:
            print(f"No runs matching ID '{run_id}' found in {base_runs_path}.")
            print("Available run names (first part of directory name):")
            for rd in run_dirs: print(f"  {rd}")
            return
        selected_run_name = matching_runs[0]
    else:
        selected_run_name = run_dirs[-1] # Use the most recent run

    run_dir_path = os.path.join(base_runs_path, selected_run_name)
    ppo_log_dir = os.path.join(run_dir_path, "ppo_") # Assuming ppo_ subdir from original script

    print(f"--- Starting Analysis for Run: {selected_run_name} ---")
    print(f"--- Log data expected in: {ppo_log_dir} ---")

    # Create dedicated output directory for this analysis version
    analysis_output_dir = _ensure_output_dir(run_dir_path, analysis_subdir_name)
    print(f"--- Analysis results will be saved in: {analysis_output_dir} ---")

    # Load data
    all_episode_data = load_episode_data(ppo_log_dir)
    if not all_episode_data:
        print(f"Failed to load any episode data from {ppo_log_dir}. Exiting.")
        return

    print(f"Loaded data for {len(all_episode_data)} episodes.")

    # Run analyses
    analyze_overall_performance(all_episode_data, analysis_output_dir)
    analyze_action_distributions_and_policy(all_episode_data, analysis_output_dir)
    analyze_rewards_per_action(all_episode_data, analysis_output_dir)
    analyze_key_actions_details(all_episode_data, analysis_output_dir) # Add more here as needed
    generate_text_report(all_episode_data, analysis_output_dir, selected_run_name)

    print(f"\n--- Analysis Complete for Run: {selected_run_name} ---")
    print(f"--- All results saved in: {analysis_output_dir} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced analysis of RL agent rewards and policy from training logs.")
    parser.add_argument("--log_dir", default="logs", help="Base log directory (e.g., where 'runs' subdir is located).")
    parser.add_argument("--run_id", default=None, help="Specific run ID (substring) to analyze (default: latest run in 'logs/runs').")
    parser.add_argument("--analysis_name", default="reward_policy_analysis_v2", help="Name for the analysis output subdirectory.")

    args = parser.parse_args()
    main_analyzer(args.log_dir, args.run_id, args.analysis_name)
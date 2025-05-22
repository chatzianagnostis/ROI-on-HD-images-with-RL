"""
ROI Detection Environment Tester - A script to verify the ROI Detection Environment works correctly
and provides interactive testing capabilities.
"""
import os
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Import the required classes
from ROIDataset import ROIDataset
from ROIDetectionEnv import ROIDetectionEnv

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

class ROIEnvTester:
    def __init__(self, dataset_path, coco_json_path, crop_size=(640, 640)):
        """
        Initialize the environment tester
        
        Args:
            dataset_path: Path to the dataset directory
            coco_json_path: Path to the COCO annotations
            crop_size: Size of the crop window (width, height)
        """
        # Create output directory
        self.output_dir = Path("env_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize the dataset
        print("Loading dataset...")
        self.dataset = ROIDataset(
            dataset_path=dataset_path,
            coco_json_path=coco_json_path,
            image_size=(640, 640),
            annotations_format="coco",
            shuffle=True
        )
        
        # Initialize the environment
        print("Setting up environment with K-means based ROI optimization...")
        self.env = ROIDetectionEnv(
            dataset=self.dataset,
            crop_size=crop_size,
            time_limit=300  # Longer time limit for testing
        )
        
        # Window name for display
        self.window_name = "ROI Environment Test"
        
        # Action history
        self.action_history = []
        self.reward_history = []

        self.action_names = [
            "Move Up (W)", "Move Down (S)", "Move Left (A)", "Move Right (D)",
            "Place Bbox (Space)", "Remove Bbox (R)", "End Episode (E)"
        ]
        
        print(f"Environment initialized with {len(self.dataset)} samples")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space keys: {list(self.env.observation_space.spaces.keys())}")
    
    def run_manual_episode(self):
        """
        Run a manual episode where user can control the actions.
        """
        print("\nStarting manual episode...")
        print("Controls:")
        print("  W/Up: Move bounding box up")
        print("  S/Down: Move bounding box down")
        print("  A/Left: Move bounding box left")
        print("  D/Right: Move bounding box right")
        print("  Space: Place bounding box")
        print("  R: Remove last bounding box")
        print("  E: End episode")
        print("  V: Visualize reward landscape")
        print("  Q: Quit")
        
        # Reset environment
        try:
            obs = self.env.reset()
            print(f"Environment reset successful. Observation keys: {list(obs.keys())}")
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return
        
        # Get initial action mask
        action_mask = self.env.action_masks()
        done = False
        total_reward = 0
        step_count = 0
        
        # Display initial state
        self._display_current_state(obs, action_mask, step_count, total_reward)
        
        while not done:
            # Wait for user input
            key = cv2.waitKey(0) & 0xFF
            
            # Map key to action
            potential_action = self._map_key_to_action(key)
            
            # Handle special keys
            if key == ord('v'):  # Visualize reward landscape
                self._visualize_reward_landscape()
                continue
            elif key == ord('q'):  # Quit
                print("Quitting...")
                break
            elif potential_action is None:
                print("Invalid key pressed. Use W/A/S/D/Space/R/E/V/Q")
                continue
            
            # Execute action
            try:
                action_name = self.action_names[potential_action]
                
                # Check if action is valid (though all actions are valid in this env)
                if action_mask[potential_action] == 1:
                    print(f"\nStep {step_count + 1}: Performing action: {action_name}")
                    
                    # Step environment
                    obs, reward, done, info = self.env.step(potential_action)
                    
                    # Update action mask for next step
                    action_mask = self.env.action_masks()
                    
                    # Update tracking
                    total_reward += reward
                    step_count += 1
                    self.action_history.append(potential_action)
                    self.reward_history.append(reward)
                    
                    print(f"Reward: {reward:.4f}, Total Reward: {total_reward:.4f}")
                    
                    # Display current state
                    self._display_current_state(obs, action_mask, step_count, total_reward)
                    
                    # Check if episode ended
                    if done:
                        self._display_episode_summary(total_reward, step_count, info)
                        break
                        
                else:
                    print(f"Action: {action_name} is INVALID. Please choose a valid action.")
                    self._show_invalid_action_message(action_name)
                    
            except Exception as e:
                print(f"Error executing action: {e}")
                break
        
        # Clean up
        cv2.destroyAllWindows()
    
    def _map_key_to_action(self, key):
        """Map keyboard input to action"""
        key_to_action = {
            ord('w'): 0,  # Move up
            ord('s'): 1,  # Move down
            ord('a'): 2,  # Move left
            ord('d'): 3,  # Move right
            ord(' '): 4,  # Place bbox (space)
            ord('r'): 5,  # Remove bbox
            ord('e'): 6,  # End episode
        }
        return key_to_action.get(key)
    
    def _display_current_state(self, obs, action_mask, step_count, total_reward):
        """Display the current state of the environment"""
        try:
            # Render the environment
            frame = self.env.render(mode='rgb_array')
            
            if frame is None:
                print("Warning: Failed to render frame")
                return
            
            # Add text overlay with current information
            frame_with_info = frame.copy()
            
            # Add step and reward info
            info_text = [
                f"Step: {step_count}",
                f"Total Reward: {total_reward:.2f}",
                f"Placed ROIs: {len(self.env.bboxes)}",
                f"Optimal ROIs: {len(self.env.optimal_rois) if self.env.optimal_rois else 0}"
            ]
            
            # Draw text on frame
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame_with_info, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_with_info, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Display valid actions
            print("\n--- Current State ---")
            print("Valid actions:")
            for i, is_valid in enumerate(action_mask):
                valid_text = "(VALID)" if is_valid else "(INVALID)"
                print(f"  - {self.action_names[i]}: {valid_text}")
            
            cv2.imshow(self.window_name, frame_with_info)
            
        except Exception as e:
            print(f"Error displaying current state: {e}")
    
    def _show_invalid_action_message(self, action_name):
        """Show invalid action message on screen"""
        try:
            frame = self.env.render(mode='rgb_array')
            if frame is not None:
                frame_with_msg = frame.copy()
                cv2.putText(frame_with_msg, f"Invalid Action: {action_name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, frame_with_msg)
        except Exception as e:
            print(f"Error showing invalid action message: {e}")
    
    def _visualize_reward_landscape(self):
        """Visualize the reward landscape"""
        try:
            print("Generating reward landscape visualization...")
            output_path = str(self.output_dir / "current_reward_landscape.jpg")
            
            result_path = self.env.visualize_reward_landscape(output_path=output_path)
            
            if result_path and os.path.exists(result_path):
                reward_landscape_img = cv2.imread(result_path)
                if reward_landscape_img is not None:
                    cv2.imshow("Reward Landscape", reward_landscape_img)
                    print(f"Reward landscape saved to: {result_path}")
                else:
                    print(f"Failed to load reward landscape image from {result_path}")
            else:
                print("Failed to generate reward landscape visualization.")
                
        except Exception as e:
            print(f"Error visualizing reward landscape: {e}")
    
    def _display_episode_summary(self, total_reward, step_count, info):
        """Display episode summary"""
        print("\n" + "="*60)
        print("EPISODE FINISHED!")
        print("="*60)
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.4f}")
        
        if 'metrics' in info:
            metrics = info['metrics']
            print("\nFinal Metrics:")
            
            if 'coverage_score' in metrics:
                print(f"  Coverage score: {metrics['coverage_score']:.4f}")
            if 'roi_matching_score' in metrics:
                print(f"  ROI matching score: {metrics['roi_matching_score']:.4f}")
            if 'optimal_count' in metrics and 'placed_count' in metrics:
                print(f"  Optimal ROI count: {metrics['optimal_count']}")
                print(f"  Placed ROI count: {metrics['placed_count']}")
            if 'time_elapsed_at_done' in metrics:
                print(f"  Time used: {metrics['time_elapsed_at_done']:.2f}s / {metrics.get('time_limit_episode', 'N/A'):.2f}s")
        
        # Save final frame
        try:
            final_frame = self.env.render(mode='rgb_array')
            if final_frame is not None:
                filename = f"final_state_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), final_frame)
                print(f"Final state saved to {filepath}")
        except Exception as e:
            print(f"Error saving final frame: {e}")
    
    def run_random_episode(self, num_steps=100, render=True):
        """
        Run an episode with random valid actions
        
        Args:
            num_steps: Maximum number of steps to run
            render: Whether to render the environment
        """
        print("\nStarting random episode...")
        
        try:
            obs = self.env.reset()
            print(f"Environment reset successful for random episode")
        except Exception as e:
            print(f"Error resetting environment for random episode: {e}")
            return
        
        done = False
        total_reward = 0
        step_count = 0
        
        if render:
            self._display_current_state(obs, self.env.action_masks(), step_count, total_reward)
        
        while not done and step_count < num_steps:
            try:
                # Get valid actions
                action_mask = self.env.action_masks()
                valid_actions = np.where(action_mask == 1)[0]
                
                if len(valid_actions) == 0:
                    print("No valid actions available. Ending episode.")
                    break
                
                # Choose random valid action
                action = np.random.choice(valid_actions)
                action_name = self.action_names[action]
                
                # Execute action
                obs, reward, done, info = self.env.step(action)
                
                # Update tracking
                total_reward += reward
                step_count += 1
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                print(f"Step {step_count}: Action: {action_name}, Reward: {reward:.4f}")
                
                if render:
                    self._display_current_state(obs, self.env.action_masks(), step_count, total_reward)
                    cv2.waitKey(50)  # Small delay for visualization
                    
            except Exception as e:
                print(f"Error in random episode step {step_count}: {e}")
                break
        
        print("\nRandom episode finished!")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.4f}")
        
        if 'metrics' in info:
            self._display_episode_summary(total_reward, step_count, info)
        
        if render:
            cv2.destroyAllWindows()

    def analyze_action_distribution(self):
        """Analyze the distribution of actions and rewards"""
        if not self.action_history:
            print("No actions recorded yet")
            return
        
        print("\nAction Distribution:")
        action_counts = np.bincount(self.action_history, minlength=self.env.action_space.n)
        
        for i, count in enumerate(action_counts):
            percentage = (count/len(self.action_history)*100) if self.action_history else 0
            print(f"  {self.action_names[i]}: {count} ({percentage:.1f}%)")
        
        print("\nReward Statistics:")
        if self.reward_history:
            print(f"  Total Reward: {sum(self.reward_history):.4f}")
            print(f"  Average Reward: {np.mean(self.reward_history):.4f}")
            print(f"  Min Reward: {min(self.reward_history):.4f}")
            print(f"  Max Reward: {max(self.reward_history):.4f}")
            
            # Reward by action
            rewards_by_action = [[] for _ in range(self.env.action_space.n)]
            for action, reward in zip(self.action_history, self.reward_history):
                rewards_by_action[action].append(reward)
            
            print("\nAverage Reward by Action:")
            for i, rewards in enumerate(rewards_by_action):
                if rewards:
                    print(f"  {self.action_names[i]}: {np.mean(rewards):.4f} (count: {len(rewards)})")
                else:
                    print(f"  {self.action_names[i]}: N/A")
        else:
            print("  No rewards recorded.")

def main():
    parser = argparse.ArgumentParser(description="ROI Detection Environment Interactive Tester")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to the dataset directory")
    parser.add_argument("--annotations", type=str, required=True,
                      help="Path to the COCO JSON annotation file")
    parser.add_argument("--crop_width", type=int, default=640,
                      help="Width of the crop window (default: 640)")
    parser.add_argument("--crop_height", type=int, default=640,
                      help="Height of the crop window (default: 640)")
    parser.add_argument("--mode", type=str, choices=['manual', 'random', 'both'], default='manual',
                      help="Test mode: manual, random, or both (default: manual)")
    
    args = parser.parse_args()
    
    try:
        tester = ROIEnvTester(
            dataset_path=args.dataset,
            coco_json_path=args.annotations,
            crop_size=(args.crop_width, args.crop_height)
        )
        
        if args.mode == 'manual' or args.mode == 'both':
            tester.run_manual_episode()
        
        if args.mode == 'random' or args.mode == 'both':
            tester.run_random_episode(num_steps=50, render=True)
        
        tester.analyze_action_distribution()
        
    except Exception as e:
        print(f"Error initializing tester: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
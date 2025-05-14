"""
ROI Detection Environment Tester - A script to verify the K-means based ROIDetectionEnv works correctly
and interacts with action masking.
"""
import os
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Import the required classes
from ROIDataset import ROIDataset  #
from ROIDetectionEnv import ROIDetectionEnv #

os.environ['LOKY_MAX_CPU_COUNT'] = '4' #

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
        self.output_dir = Path("env_test_results") #
        self.output_dir.mkdir(exist_ok=True) #
        
        # Initialize the dataset
        print("Loading dataset...") #
        self.dataset = ROIDataset( #
            dataset_path=dataset_path, #
            coco_json_path=coco_json_path, #
            image_size=(640, 640), #
            annotations_format="coco", #
            shuffle=True #
        )
        
        # Initialize the environment
        print("Setting up environment with K-means based ROI optimization...") #
        self.env = ROIDetectionEnv( #
            dataset=self.dataset, #
            crop_size=crop_size, #
            time_limit=300  # Longer time limit for testing #
        )
        
        # Window name for display
        self.window_name = "ROI Environment Test" #
        
        # Action history
        self.action_history = [] #
        self.reward_history = [] #

        self.action_names = [ # Define action names for display
            "Move Up (W)", "Move Down (S)", "Move Left (A)", "Move Right (D)",
            "Place Bbox (Space)", "Remove Bbox (R)", "End Episode (E)"
        ]
        
        print(f"Environment initialized with {len(self.dataset)} samples") #
        print(f"Action space: {self.env.action_space}") #
        print(f"Observation space: {self.env.observation_space}") #
    
    def run_manual_episode(self):
        """
        Run a manual episode where user can control the actions, respecting action masks.
        """
        print("\nStarting manual episode...") #
        print("Controls:") #
        print("  W/Up: Move bounding box up") #
        print("  S/Down: Move bounding box down") #
        print("  A/Left: Move bounding box left") #
        print("  D/Right: Move bounding box right") #
        print("  Space: Place bounding box") #
        print("  R: Remove last bounding box") #
        print("  E: End episode") #
        print("  V: Visualize reward landscape")
        print("  K: Toggle optimal ROIs visibility (effect may vary after first step due to env.render)") #
        print("  Q: Quit") #
        
        # Reset environment
        obs = self.env.reset() #
        action_mask = self.env.action_masks() # Get initial action mask
        done = False #
        total_reward = 0 #
        
        # Toggle for showing optimal ROIs
        show_optimal = True #
        
        # Convert dict observation to image for display
        frame = obs['image'].copy() #
        
        # Draw optimal ROIs if available for the initial frame
        if show_optimal and self.env.optimal_rois: #
            for roi in self.env.optimal_rois: #
                cv2.rectangle(frame,  #
                             (int(roi[0]), int(roi[1])),  #
                             (int(roi[0] + roi[2]), int(roi[1] + roi[3])), #
                             (0, 0, 255), 1)  # Red color for optimal ROIs #
                             
        cv2.imshow(self.window_name, frame) #
        
        while not done:
            # Display current action mask information
            print("\n--- Current State ---")
            print("Valid actions (based on current mask):")
            for i, is_valid in enumerate(action_mask):
                valid_text = "(VALID)" if is_valid else "(INVALID)"
                print(f"  - {self.action_names[i]}: {valid_text}")

            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key for manual control #
            
            # Map key to action
            potential_action = None #
            
            if key == ord('w'): # W key #
                potential_action = 0  # Move up #
            elif key == ord('s'): # S key #
                potential_action = 1  # Move down #
            elif key == ord('a'): # A key #
                potential_action = 2  # Move left #
            elif key == ord('d'): # D key #
                potential_action = 3  # Move right #
            elif key == ord(' '): # Space #
                potential_action = 4  # Place bbox #
            elif key == ord('r'): # R #
                potential_action = 5  # Remove bbox #
            elif key == ord('e'): # E #
                potential_action = 6  # End episode #
            elif key == ord('v'): # V key #
                print("Visualizing reward landscape...")
                visualization_path = self.env.visualize_reward_landscape( #
                    str(self.output_dir / "current_reward_landscape.jpg")
                )
                if visualization_path:
                    reward_landscape_img = cv2.imread(visualization_path) #
                    if reward_landscape_img is not None:
                        cv2.imshow("Reward Landscape", reward_landscape_img) #
                    else:
                        print(f"Failed to load reward landscape image from {visualization_path}")
                else:
                     print("Failed to generate reward landscape visualization.")
                continue # Continue to next key press without stepping env
            elif key == ord('k'): # K - toggle optimal ROIs #
                show_optimal = not show_optimal #
                # Re-render the current view based on the new 'show_optimal' state for immediate feedback
                # This part is tricky because env.render() might always draw them.
                # For simplicity, we'll rely on the next full render after an action.
                # Or, if no action is taken, the display won't update until an action is taken.
                # Let's try to update the current frame.
                # This will re-render using the env.render() output which already includes optimal ROIs by default.
                # The 'show_optimal' toggle in the visualizer is thus somewhat limited in its effect after the first frame.
                temp_frame = self.env.render(mode='rgb_array') #
                # If we want 'show_optimal' to truly turn them off, env.render() would need a flag,
                # or we'd reconstruct the frame here.
                # For now, accept env.render()'s behavior.
                if not show_optimal:
                     print("Optimal ROIs are nominally hidden (but env.render might still show them).")
                else:
                     print("Optimal ROIs are shown.")
                cv2.imshow(self.window_name, temp_frame) #
                continue #
            elif key == ord('q'): # Q #
                break #
                
            if potential_action is not None:
                action_name_selected = self.action_names[potential_action]
                if action_mask[potential_action] == 1: # Check action validity
                    print(f"Performing action: {action_name_selected}")
                    
                    # Step environment
                    obs, reward, done, info = self.env.step(potential_action) #
                    action_mask = info.get('action_mask', np.ones(self.env.action_space.n, dtype=np.int8)) # Update mask for next state

                    # Display reward
                    print(f"Reward: {reward:.4f}") #
                    total_reward += reward #
                    
                    # Record action and reward
                    self.action_history.append(potential_action) #
                    self.reward_history.append(reward) #
                    
                    # Render the environment
                    frame_to_display = self.env.render(mode='rgb_array') #
                    if frame_to_display is not None: #
                        # The env.render() now handles drawing optimal ROIs if they exist.
                        # The 'show_optimal' flag here has limited effect on what env.render() does by default.
                        cv2.imshow(self.window_name, frame_to_display) #
                else:
                    print(f"Action: {action_name_selected} is INVALID. Please choose a valid action.")
                    # Optionally, update the display to show the invalid message on screen
                    frame_with_msg = self.env.render(mode='rgb_array').copy()
                    cv2.putText(frame_with_msg, f"Invalid Action: {action_name_selected}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(self.window_name, frame_with_msg)
                    continue # Skip to next iteration of while loop

                # Check if episode ended
                if done: #
                    print("\nEpisode finished!") #
                    print(f"Total reward: {total_reward:.4f}") #
                    if 'metrics' in info: #
                        metrics = info['metrics'] #
                        print("\nFinal Metrics:") #
                        
                        if 'coverage_score' in metrics: #
                            print(f"  Coverage score: {metrics['coverage_score']:.4f}") #
                        if 'roi_matching_score' in metrics: #
                            print(f"  ROI matching score: {metrics['roi_matching_score']:.4f}") #
                        # Note: 'efficiency_score' and 'overlap_penalty' were present in the original snippet but might not be in all env versions
                        if 'efficiency_score' in metrics: 
                            print(f"  Efficiency score: {metrics['efficiency_score']:.4f}")
                        if 'overlap_penalty' in metrics:
                            print(f"  Overlap penalty: {metrics['overlap_penalty']:.4f}")
                        if 'optimal_count' in metrics and 'placed_count' in metrics: #
                            print(f"  Optimal ROI count: {metrics['optimal_count']}") #
                            print(f"  Placed ROI count: {metrics['placed_count']}") #
                        
                        # Original snippet had 'time_info = metrics['time']' - check if 'time' or direct time metrics are in your env
                        if 'time_elapsed_at_done' in metrics:
                             print(f"  Time used: {metrics['time_elapsed_at_done']:.2f}s / {metrics.get('time_limit_episode', 'N/A'):.2f}s")

                    # Save final frame
                    final_frame_to_save = self.env.render(mode='rgb_array') #
                    if final_frame_to_save is not None: #
                        filename = f"final_state_{time.strftime('%Y%m%d_%H%M%S')}.jpg" #
                        filepath = self.output_dir / filename #
                        cv2.imwrite(str(filepath), final_frame_to_save) #
                        print(f"Final state saved to {filepath}") #
        
        # Close all windows
        cv2.destroyAllWindows() #
    
    # run_random_episode and analyze_action_distribution remain unchanged unless specifically requested.
    # For run_random_episode to use action masking correctly, it would also need to
    # get and use the action_mask to sample only from valid actions.

    def run_random_episode(self, num_steps=100, render=True): #
        """
        Run an episode with random valid actions
        
        Args:
            num_steps: Maximum number of steps to run
            render: Whether to render the environment
        """
        print("\nStarting random episode with action masking...") #
        
        obs = self.env.reset() #
        action_mask = self.env.action_masks()
        done = False #
        total_reward = 0 #
        step_count = 0 #
        
        if render: #
            frame = self.env.render(mode='rgb_array') #
            if frame is not None: #
                 # env.render() should handle optimal ROIs drawing
                cv2.imshow(self.window_name, frame) #
        
        while not done and step_count < num_steps: #
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) == 0:
                print("No valid actions available. Ending episode.")
                # Potentially force end action if it's valid, or just break
                if action_mask[6] == 1: # Action 6 is End Episode
                    action = 6
                else: # Should not happen in a well-designed env unless it's a terminal state already
                    break 
            else:
                action = np.random.choice(valid_actions)
            
            obs, reward, done, info = self.env.step(action) #
            action_mask = info.get('action_mask', np.ones(self.env.action_space.n, dtype=np.int8))
            
            self.action_history.append(action) #
            self.reward_history.append(reward) #
            total_reward += reward #
            step_count += 1 #
            
            action_name_performed = self.action_names[action] #
            print(f"Step {step_count}: Action: {action_name_performed}, Reward: {reward:.4f}") #
            
            if render: #
                frame = self.env.render(mode='rgb_array') #
                if frame is not None: #
                    cv2.imshow(self.window_name, frame) #
                    cv2.waitKey(50) # Small delay, was 1 #
        
        print("\nEpisode finished!") #
        print(f"Total steps: {step_count}") #
        print(f"Total reward: {total_reward:.4f}") #
        
        if 'metrics' in info: #
            # ... (metric printing logic remains similar to run_manual_episode's end) ...
            metrics = info['metrics'] #
            print("\nFinal Metrics:") #
            if 'coverage_score' in metrics: print(f"  Coverage score: {metrics['coverage_score']:.4f}") #
            if 'roi_matching_score' in metrics: print(f"  ROI matching score: {metrics['roi_matching_score']:.4f}") #
            if 'optimal_count' in metrics and 'placed_count' in metrics: #
                print(f"  Optimal ROI count: {metrics['optimal_count']}") #
                print(f"  Placed ROI count: {metrics['placed_count']}") #
            if 'time_elapsed_at_done' in metrics:
                 print(f"  Time used: {metrics['time_elapsed_at_done']:.2f}s / {metrics.get('time_limit_episode', 'N/A'):.2f}s")
        
        final_frame_to_save_random = self.env.render(mode='rgb_array') #
        if render and final_frame_to_save_random is not None: #
            filename = f"random_episode_{time.strftime('%Y%m%d_%H%M%S')}.jpg" #
            filepath = self.output_dir / filename #
            cv2.imwrite(str(filepath), final_frame_to_save_random) #
            print(f"Final state saved to {filepath}") #
        
        if render: #
            cv2.destroyAllWindows() #

    def analyze_action_distribution(self): #
        """Analyze the distribution of actions and rewards"""
        if not self.action_history: #
            print("No actions recorded yet") #
            return #
        
        print("\nAction Distribution:") #
        action_counts = np.bincount(self.action_history, minlength=self.env.action_space.n) #
        
        for i, count in enumerate(action_counts): #
            print(f"  {self.action_names[i]}: {count} ({count/len(self.action_history)*100:.1f}%)") #
        
        print("\nReward Statistics:") #
        print(f"  Total Reward: {sum(self.reward_history):.4f}") #
        if self.reward_history:
            print(f"  Average Reward: {np.mean(self.reward_history):.4f}") #
            print(f"  Min Reward: {min(self.reward_history):.4f}") #
            print(f"  Max Reward: {max(self.reward_history):.4f}") #
        else:
            print("  No rewards recorded.")

        rewards_by_action = [[] for _ in range(self.env.action_space.n)] #
        for action, reward in zip(self.action_history, self.reward_history): #
            rewards_by_action[action].append(reward) #
        
        print("\nAverage Reward by Action:") #
        for i, rewards in enumerate(rewards_by_action): #
            if rewards: #
                print(f"  {self.action_names[i]}: {np.mean(rewards):.4f}") #
            else: #
                print(f"  {self.action_names[i]}: N/A") #

def main(): #
    parser = argparse.ArgumentParser(description="ROI Detection Environment Tester with Action Masking") #
    parser.add_argument("--dataset", type=str, required=True, #
                      help="Path to the dataset directory") #
    parser.add_argument("--annotations", type=str, required=True, #
                      help="Path to the COCO JSON annotation file") #
    parser.add_argument("--crop_width", type=int, default=640, #
                      help="Width of the crop window (default: 640)") #
    parser.add_argument("--crop_height", type=int, default=640, #
                      help="Height of the crop window (default: 640)") #
    parser.add_argument("--mode", type=str, choices=['manual', 'random', 'both'], default='manual', #
                      help="Test mode: manual, random, or both (default: manual)") #
    
    args = parser.parse_args() #
    
    tester = ROIEnvTester( #
        dataset_path=args.dataset, #
        coco_json_path=args.annotations, #
        crop_size=(args.crop_width, args.crop_height) #
    )
    
    if args.mode == 'manual' or args.mode == 'both': #
        tester.run_manual_episode() #
    
    if args.mode == 'random' or args.mode == 'both': #
        tester.run_random_episode(num_steps=50, render=True) # # Added render=True for consistency
    
    tester.analyze_action_distribution() #

if __name__ == "__main__": #
    main() #
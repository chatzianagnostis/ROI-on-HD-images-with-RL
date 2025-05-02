from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class DynamicRewardShapingCallback(BaseCallback):
    """
    Callback for adjusting reward shaping based on agent performance.
    """
    def __init__(self, check_freq=15000, window_size=1000, initial_coeff=0.01, 
                 boost_coeff=0.05, verbose=2):
        super(DynamicRewardShapingCallback, self).__init__(verbose)
        self.check_freq = check_freq  # How often to check performance
        self.window_size = window_size  # Size of reward window to evaluate
        self.initial_coeff = initial_coeff # Default shaping coefficient
        self.boost_coeff = boost_coeff  # Boosted shaping coefficient
        self.rewards = []  # Track episode rewards
        self.shaping_active = False  # Track if boosted shaping is active
        self.last_mean = None  # Last mean reward
        
    def _on_step(self) -> bool:
        # Get current episode reward
        if self.locals.get("dones")[0]:
            ep_info = self.locals.get("infos")[0].get("episode")
            if ep_info:
                ep_reward = ep_info["r"]
                self.rewards.append(ep_reward)
                
                # Limit window size
                if len(self.rewards) > self.window_size:
                    self.rewards.pop(0)
        
        # Check if it's time to evaluate
        if self.num_timesteps % self.check_freq == 0 and len(self.rewards) > 0:
            current_mean = np.mean(self.rewards)
            
            # First check or detecting a plateau
            if self.last_mean is None or (self.last_mean > 0 and 
                                         current_mean < self.last_mean * 1.02):
                # Agent is plateauing or performance is decreasing
                self.shaping_active = True
                # Update environment's shaping coefficient
                env = self.training_env.envs[0]
                env.shaping_coeff = self.boost_coeff
                
                if self.verbose > 0:
                    print(f"Timestep {self.num_timesteps}: Activating boosted reward shaping")
            else:
                # Agent is improving, use normal shaping
                self.shaping_active = False
                env = self.training_env.envs[0]
                env.shaping_coeff = self.initial_coeff
                
                if self.verbose > 0:
                    print(f"Timestep {self.num_timesteps}: Reverting to normal reward shaping")
            
            self.last_mean = current_mean
            
        return True

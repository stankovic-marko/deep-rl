#!/bin/python
from stable_baselines3.common.callbacks import BaseCallback
import os
from gym_env import Bombarder
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        self.last_saved = 0
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls - self.last_saved >= self.save_freq:
            model_path = os.path.join(
                self.save_path, f'model_new_obs{self.n_calls}_steps')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
            self.last_saved = self.n_calls
        return True


def make_env():
    def _init():
        return Bombarder(render_mode="human")
    return _init


if __name__ == "__main__":
    # Number of environments to run in parallel
    num_envs = 16
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])

    save_freq = 100000
    save_path = './models_bomberman/'
    callback = SaveOnBestTrainingRewardCallback(
        save_freq, save_path)
    model = PPO("MlpPolicy", envs, verbose=2, batch_size=128)
    model.learn(total_timesteps=10000000, progress_bar=True,
                callback=callback)
    # model.save("ppo_bombarder2")

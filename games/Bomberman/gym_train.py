#!/bin/python
from stable_baselines3.common.callbacks import BaseCallback
import os
from gym_env import Bombarder
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import csv
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LearningCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    """

    def __init__(self, save_freq: int, save_path: str, logs_path: str = "logs", log_filename: str = "env_log.csv", verbose: int = 1):
        super(LearningCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.logs_path = logs_path
        self.log_filename = log_filename
        if self.logs_path is not None:
            os.makedirs(self.logs_path, exist_ok=True)
        self.csv_file = open(os.path.join(
            self.logs_path, self.log_filename), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestep', 'Score', 'Won', 'Steps Alive'])

    def _init_callback(self) -> None:
        # Create folder if needed
        self.last_saved = 0
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # If first environment is done
        if self.locals['dones'][0]:
            timestep = self.num_timesteps
            score = self.locals['infos'][0].get("score")
            won = self.locals['infos'][0].get("won")
            steps_alive = self.locals['infos'][0].get("steps_alive")
            # Log timestep and score
            self.csv_writer.writerow([timestep, score, won, steps_alive])
            self.csv_file.flush()

        if self.n_calls - self.last_saved >= self.save_freq:
            model_path = os.path.join(
                self.save_path, f'model_new_obs{self.n_calls}_steps')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
            self.last_saved = self.n_calls
        return True


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Extract dimensions from observation space
        n_input_channels = observation_space.shape[0]
        print(
            observation_space.shape[0], observation_space.shape[1], observation_space.shape[2])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size of the output of the last convolutional layer
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(
                observation_space.sample()[None]).float()).shape[1]

        # Define the final fully connected layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


def make_env():
    def _init():
        return Bombarder(render_mode="human")
    return _init


if __name__ == "__main__":
    # Number of environments to run in parallel
    num_envs = 8
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])

    save_freq = 30000
    save_path = './models_bomberman_hal/'
    callback = LearningCallback(
        save_freq, save_path)
    model = PPO("CnnPolicy", envs, verbose=2,
                batch_size=512, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10000000, progress_bar=True,
                callback=callback)
    model.save("ppo_bombarder_hal")

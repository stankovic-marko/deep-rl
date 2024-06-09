import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv


# Define a custom CNN for feature extraction


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64, features_dim),
            nn.ReLU()


        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(
                observation_space.sample()[None]).float()).shape[1]

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

# # Create the Mario Bros environment
# # Replace with your custom Mario Bros environment
# env = gym.make('ALE/MarioBros-v5', render_mode='human')
# env = DummyVecEnv([lambda: env])

# # Train the PPO agent
# model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=100000)

# # # Evaluate the trained agent
# # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# # print(f'Mean reward: {mean_reward} +/- {std_reward}')

# # Save the model
# model.save("ppo_mario_bros_rgb")

# # Load the model
# model = PPO.load("ppo_mario_bros_rgb")


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

    def _init_callback(self) -> None:
        # Create folder if needed
        self.last_saved = 0
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.logs_path is not None:
            os.makedirs(self.logs_path, exist_ok=True)
        self.csv_file = open(os.path.join(
            self.logs_path, self.log_filename), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestep', 'Score'])

    def _on_step(self) -> bool:
        # If first environment is done
        if self.locals['dones'][0]:
            timestep = self.num_timesteps
            print(self.locals['rewards'][0])
            score = self.locals['infos'][0].get("rewards")
            # Log timestep and score
            self.csv_writer.writerow([timestep, score])
            self.csv_file.flush()

        if self.n_calls - self.last_saved >= self.save_freq:
            model_path = os.path.join(
                self.save_path, f'model_with_discrete_{self.n_calls}_steps')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
            self.last_saved = self.n_calls
        return True


def make_env():
    def _init():
        return gymnasium.make('ALE/MarioBros-v5', render_mode='human')
    return _init


if __name__ == "__main__":
    # Number of environments to run in parallel
    num_envs = 4
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])

    save_freq = 100000
    save_path = './models_mario/'
    callback = LearningCallback(
        save_freq, save_path)
    model = PPO("CnnPolicy", envs, verbose=2,
                batch_size=128, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10000000, progress_bar=True,
                callback=callback)
    # model.save("ppo_bombarder2")

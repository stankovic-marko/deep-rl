from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import BaseCallback
import flappy_bird_gymnasium
import gymnasium
import csv


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
        self.csv_writer.writerow(['Timestep', 'Max', 'Avg'])

    def _on_step(self) -> bool:
        # If first environment is done
        envs = {}
        for i, d in enumerate(self.locals["dones"]):
            if d:
                score = self.locals['infos'][i].get("score")
                envs[i] = score
        sum = 0.0
        max = 0.0
        avg = 0.0
        if len(envs) > 0:
            for i in envs:
                sum += envs[i]
                if envs[i] > max:
                    max = envs[i]
            avg = sum / len(envs)
            timestep = self.num_timesteps
            self.csv_writer.writerow([timestep, max, avg])
            self.csv_file.flush()

        # if self.locals['dones'][0]:
        #     timestep = self.num_timesteps
        #     score = self.locals['infos'][0].get("score")
        #     # Log timestep and score
        #     self.csv_writer.writerow([timestep, score])
        #     self.csv_file.flush()

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
        return gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    return _init


if __name__ == "__main__":
    # Number of environments to run in parallel
    num_envs = 20
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])

    save_freq = 100000
    save_path = './models_flappy/'
    logs_path = "./logs_third/"
    callback = LearningCallback(
        save_freq, save_path, logs_path=logs_path)
    model = PPO("MlpPolicy", envs, verbose=2, batch_size=128)
    model.learn(total_timesteps=10000000, progress_bar=True,
                callback=callback)
    # model.save("ppo_bombarder2")

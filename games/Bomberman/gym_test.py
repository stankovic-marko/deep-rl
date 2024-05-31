#!/bin/python
from gym_env import Bombarder
from stable_baselines3 import PPO
import gymnasium
env = Bombarder(render_mode="human")


model = PPO("MlpPolicy", env, verbose=2, batch_size=128)
model.learn(total_timesteps=10000, progress_bar=True)
model.save("ppo_bombarder")

# model.load("ppo_bombarder")
# vec_env = model.get_env()
# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     #print(obs, reward)
#     vec_env.render()
#     # VecEnv resets automatically
#     # if done:
#     #   obs = vec_env.reset()

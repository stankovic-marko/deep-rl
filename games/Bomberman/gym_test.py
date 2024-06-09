#!/bin/python
from gym_env import Bombarder
from stable_baselines3 import PPO
import gymnasium
env = Bombarder(render_mode="human")


#model = PPO("MlpPolicy", env, verbose=2, batch_size=128,device='cpu')


model = PPO.load("./models_bomberman_hal/model_new_obs60000_steps", env=env)
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    print(action, reward)
    #print(obs, reward)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

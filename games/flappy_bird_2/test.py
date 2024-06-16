#!/bin/python
from stable_baselines3 import PPO
import gymnasium
import flappy_bird_gymnasium
env = gymnasium.make('FlappyBird-v0', render_mode='human', use_lidar=True)


#model = PPO("MlpPolicy", env, verbose=2, batch_size=128,device='cpu')


model = PPO.load(
    "./models_flappy/model_with_discrete_500000_steps", env=env)
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    #print(obs, reward)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

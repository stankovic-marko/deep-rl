import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["figure.autolayout"] = True
columns = ["Timestep", "Steps Alive"]
df = pd.read_csv("./logs/env_log.csv", usecols=columns)
plt.plot(df.Timestep, df["Steps Alive"])
plt.show()

columns = ["Timestep", "Score"]
df = pd.read_csv("./logs/env_log.csv", usecols=columns)
plt.plot(df.Timestep, df.Score)
plt.show()

columns = ["Timestep", "Won"]
df = pd.read_csv("./logs/env_log.csv", usecols=columns)
plt.plot(df.Timestep, df.Won)
plt.show()

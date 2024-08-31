import matplotlib.pyplot as plt
import pandas as pd

df_rgb = pd.read_csv('./results/rgb.csv')
df_fft = pd.read_csv('./results/fft.csv')
fps = 30
df_rgb['time'] = df_rgb['frame'] / fps

plt.plot(df_rgb['time'], df_rgb['rgb'])
plt.xlabel("Time [s]")
plt.ylabel("RGB Signal")
plt.savefig('rgb.png')
import matplotlib.pyplot as plt
import pandas as pd
import os
path_base = os.path.join('C:/Users','Dell','studia-pliki-robocze', 'magisterka', 'rezultaty')
df_rgb = pd.read_csv(os.path.join(path_base, 'rgb_pos.csv'))
df_fft = pd.read_csv(os.path.join(path_base, 'fft_pos.csv'))
fps = 30
df_rgb['time'] = df_rgb['frame'] / fps
num = 1
plt.subplot(2, 2, num)
plt.plot(df_rgb['time'], df_rgb['rgb'])
plt.xlabel("Time [s]")
plt.ylabel("G Channel Signal")

num+=1
plt.subplot(2, 2, num)
plt.plot(df_rgb['time'], df_rgb['rgb_detrended'])
plt.xlabel("Time [s]")
plt.ylabel("G Channel Detrended Signal")

num+=1
plt.subplot(2, 2, num)
plt.plot(df_rgb['time'], df_rgb['signal_filtered'])
plt.xlabel("Time [s]")
plt.ylabel("POS Filtered Signal")

num+=1
plt.subplot(2, 2, num)
plt.plot(df_fft['freq'], df_fft['sig'])
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
#plt.savefig('rgb.png')
plt.show()
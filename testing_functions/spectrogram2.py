# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import mne
from mne.time_frequency import tfr_array_multitaper
import cv2
from datetime import datetime

raw = mne.io.read_raw_edf("F:/CAP Sleep Database/ins1.edf", verbose = False)
original_fs = int(raw.info['sfreq'])
assert(raw["C4-A1"][1][original_fs] == 1)   # check signal if listed frequency is true
print(original_fs)


pID = "ins1"
# Load data
with pd.HDFStore("E:/HDD documents/University/Thesis/Thesis B code/data/CAP_overlap_filter2.h5", mode = 'r') as store:
    data = store[pID]

data = np.array(data)

# Multitaper spectrogram
# data = np.reshape(data1, (1, 1, 3840))
sfreq = 128
f_min = 0.1
f_max = 50.0
interval = (f_max - f_min)/224
freqs = np.linspace(f_min, f_max, num = 224, endpoint = True)

T = 30  #T = n_cycles / freq
n_cycles = freqs    # Number of cycles in wavelet per frequency interval

delta_f = 0.2
my_dpi = 290
matplotlib.rcParams['savefig.dpi'] = my_dpi
matplotlib.rcParams["figure.figsize"] = (1, 1.005)  # for terminal
#plt.rcParams['savefig.dpi'] = my_dpi
#plt.rcParams["figure.figsize"] = (1, 1.025) # for jupyter
#epoch_reshaped = epoch.reshape((epoch.shape[0], 1, epoch.shape[1]))

indices = np.arange(0, data.shape[0], 1)
#indices = np.arange(0, data.shape[0], 10)
# indices = np.arange(0, 500, 100)
time_bandwidth = 4    # 3 tapers
now = datetime.now()
dt_string_start = now.strftime("%d-%m-%Y %H.%M.%S")
x = np.arange(1, 3841, 1)
# %%
for i in indices:
    epoch = data[i][1:]
    stage = data[i][0]
    epoch = epoch.reshape((1, 1, epoch.shape[0]))
    power = tfr_array_multitaper(epoch, sfreq = sfreq, freqs = freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, output = 'power')
    
    #plt.subplot(1, 2, 1)
    # plt.rcParams["figure.figsize"] = (1, 1.025)
    
    plt.pcolormesh(x, freqs, power[0][0], shading = 'flat')
    plt.axis('off')
    plt.savefig(f"E:/HDD documents/University/Thesis/Thesis B code/testing_functions/tfr_images/{pID}-{i}-{stage}.png", bbox_inches='tight', transparent=True, pad_inches=0, dpi = my_dpi)
    # plt.plot(epoch)
    #plt.show()
    plt.clf()
    plt.close()
now = datetime.now()
dt_string_end = now.strftime("%d-%m-%Y %H.%M.%S")
print(f"Start: {dt_string_start} End: {dt_string_end}")

# %%

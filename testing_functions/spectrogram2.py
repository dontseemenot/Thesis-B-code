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




# raw = mne.io.read_raw_edf("F:/CAP Sleep Database/ins1.edf", verbose = False)
# original_fs = int(raw.info['sfreq'])
# assert(raw["C4-A1"][1][original_fs] == 1)   # check signal if listed frequency is true
# print(original_fs)

pIDs = ["ins1", "n1"]
nums = [579, 920]
for pID, num in zip(pIDs, nums):
# Load data
    with pd.HDFStore("E:/HDD documents/University/Thesis/Thesis B code/data/CAP_no_overlap_filter2/CAP_no_overlap_filter2.h5", mode = 'r') as store:
        data = store[pID]

    # Set variables
    data = np.array(data)
    sfreq = 128
    f_min = 0.1
    f_max = 25.0
    freqs = np.linspace(f_min, f_max, num = 250, endpoint = True)


    x = np.arange(1, 3841, 1)

    matplotlib.rcParams["figure.figsize"] = (10, 25)

    # ins1 579
    # n1 920
    #for i in np.arange(0, 1, 50):
    #for i in np.arange(0, data.shape[0], 1):
    epoch = data[num][1:]
    stage = data[num][0]
    plt.subplot(5, 1, 2)
    plt.plot(epoch)
    plt.ylabel("Voltage")
    plt.title("Time signal")
    
    fft = np.fft.rfft(epoch)
    absfft = np.abs(fft)
    power = np.square(absfft)
    frequency = np.linspace(0, sfreq/2, len(power))
    plt.subplot(5, 1, 1)
    plt.plot(frequency, power)
    plt.ylabel("Power")
    plt.title("FFT")
    plt.xlim([0, 40])
    epoch = epoch.reshape((1, 1, epoch.shape[0]))

    delta_fs = [1, 1, 1]
    n_cycless = [1, 2, 3]
    window = 6    # seconds
    ### User changeable variables
    for i, (delta_f, n_cycles) in enumerate(zip(delta_fs, n_cycless)):

        time_bandwidth = window * delta_f
        n_cycles_freq = freqs * n_cycles    # Number of cycles in wavelet per frequency interval;



        power = tfr_array_multitaper(epoch, sfreq = sfreq, freqs = freqs, n_cycles=n_cycles_freq, time_bandwidth=time_bandwidth, output = 'power')
        
        plt.subplot(5, 1, i + 3)
        name1 = f"{pID}-{num}-{stage}"
        name2 = f"BW={time_bandwidth}-ncycles={n_cycles}"
        #plt.rcParams["figure.figsize"] = (1, 1.025)
        plt.title(f"{name2}")
        plt.pcolormesh(x, freqs, power[0][0], shading = 'flat')
        plt.suptitle(f"{name1}")

    plt.savefig(f"E:/HDD documents/University/Thesis/Thesis B code/testing_functions/tfr_picked/6s-{name1}-{name2}.png", transparent=True, pad_inches=0, dpi = 300)
    plt.xlabel("Sample")
    plt.ylabel("Freq")

    plt.show()
    plt.clf()
    plt.close()

    


# %%

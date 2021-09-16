# %%
from re import A
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
import mne
from mne.filter import filter_data, create_filter
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter



rawF = "F:/Thesis B insomnia data/Insomnia data/Data_Study3/Berlin PSG EDF files/IM_01 I.edf"

raw = mne.io.read_raw_edf(rawF, verbose = False)
fs1 = int(raw.info['sfreq'])
data1 = raw['C4:A1'][0][0]
if fs1 < 128:
    data2 = mne.filter.resample(data1, up = 128/fs1, down = 1, verbose = False)
elif fs1 > 128:
    data2 = mne.filter.resample(data1, up = 1, down = fs1/128, verbose = False)
fs2 = 128

data3 = filter_data(data2, sfreq = fs2, l_freq = None, h_freq = 40, l_trans_bandwidth = 10, h_trans_bandwidth = 10, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
fs3 = fs2

data4 = filter_data(data2, sfreq = fs2, l_freq = 0.5, h_freq = 50, l_trans_bandwidth = 0.01, h_trans_bandwidth = 0.01, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
fs4 = fs2
# %%
a = 40
b = 70
plt.plot(data1[a*fs1:b*fs1], color = "g")
plt.show()
plt.plot(data2[a*fs2:b*fs2], color = "g")
plt.show()
plt.plot(data3[a*fs3:b*fs3], color = "g")
plt.show()
plt.plot(data4[a*fs4:b*fs4], color = "g")
# %%
for d, fs in zip([data1, data2, data3], [fs1, fs2, fs3]):
    fft = np.fft.rfft(d)
    absfft = np.abs(fft)
    power = np.square(absfft)
    frequency = np.linspace(0, fs/2, len(power))
    plt.plot(frequency, power, color = "b")
    plt.xlim(left = 0, right=60)
    plt.show()


# %%

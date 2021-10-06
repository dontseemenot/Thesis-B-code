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



rawF = "F:/Thesis B insomnia data/Insomnia data/Data_Study3/Berlin PSG EDF files/IM_03 G.edf"

raw = mne.io.read_raw_edf(rawF, verbose = False)
fs1 = int(raw.info['sfreq'])
data1 = raw['C4:A1'][0][0]
if fs1 < 128:
    data2 = mne.filter.resample(data1, up = 128/fs1, down = 1, verbose = False)
elif fs1 > 128:
    data2 = mne.filter.resample(data1, up = 1, down = fs1/128, verbose = False)
fs2 = 128
data3 = filter_data(data2, sfreq = fs2, l_freq = None, h_freq = 40, l_trans_bandwidth = None, h_trans_bandwidth = 10, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
fs3 = fs2
data4 = filter_data(data2, sfreq = fs2, l_freq = 0.5, h_freq = 40, l_trans_bandwidth = 0.5, h_trans_bandwidth = 2, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
fs4 = fs2
data5 = filter_data(data2, sfreq = fs2, l_freq = 0.5, h_freq = 40, l_trans_bandwidth = 0.1, h_trans_bandwidth = 2, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
fs5 = fs2

# %%
a = 226*30 
b = 227*30 
# plt.plot(data1[a*fs1:b*fs1], color = "g")
# plt.ylim([-0.00025, 0.00025])
# plt.show()
for d, fs, title in zip([data2, data3, data4, data5], [fs2, fs3, fs4, fs5], ["Original signal", "Signal after lowpass 40Hz with 2Hz high transition bandwidth", "Signal after bandpass 0.5-40Hz with 0.5Hz low transition bandwidth, 2Hz high transition bandwidth", "Signal after bandpass 0.5-40Hz with 0.01Hz low transition bandwidth, 2Hz high transition bandwidth"]):
    plt.rcParams["figure.figsize"] = (10,4)
    plt.plot(d[a*fs2:b*fs2], color = "g")
    plt.title(title)
    plt.ylim([-0.00008, 0.00008])
    plt.xlabel('Sample')
    plt.ylabel('Voltage')
    plt.show()

# %%
for d, fs, title in zip([data2, data3, data4, data5], [fs2, fs3, fs4, fs5], ["Original signal", "Signal after lowpass 40Hz with 2Hz high transition bandwidth", "Signal after bandpass 0.5-40Hz with 0.5Hz low transition bandwidth, 2Hz high transition bandwidth", "Signal after bandpass 0.5-40Hz with 0.01Hz low transition bandwidth, 2Hz high transition bandwidth"]):
    fft = np.fft.rfft(d[a*fs:b*fs])
    absfft = np.abs(fft)
    power = np.square(absfft)
    frequency = np.linspace(0, fs/2, len(power))
    plt.plot(frequency, power, color = "b")
    plt.xlim(left = 0, right=10)
    plt.ylim([0, 0.0002])
    plt.ylabel("Power")
    plt.xlabel("Hz")
    plt.title(title)
    plt.xticks(np.arange(0, 10, step=1))
    plt.show()


# %%

# %%
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "E:/HDD documents/University/Thesis/Thesis B code/testing_functions/blues.00000.wav"
signal, sr = librosa.load(file, sr = 22050)   # signal is 1D array
librosa.display.waveplot(signal, sr = sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
plt.clf()
# %%
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
plt.plot(frequency, magnitude)
plt.xlabel("Freq")
plt.ylabel("Magnitude")
plt.show()
# %%
n_fft = 2048    # num of samples per fft. window we are considering when performing fft.
hop_length = 512    # amount we are shifting each fourier transform to the right
stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
spectrogram =  np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, sr = sr, hop_length = hop_length)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.colorbar()
plt.show()
# %%

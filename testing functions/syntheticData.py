# %%
import os
import numpy
import pandas as pd

destDir = 'F:\\Berlin data formatted'


for x in [1, 2, 4, 5, 6]:
    print(x)
    epoch = []
    for a in range(1, 401):
        eeg = []
        for b in range(3840):
            eeg.append(float(0.8))
        eeg = numpy.array(eeg)
        epoch.append([x, 'TestData', eeg, 'I'])
    s = pd.Series(epoch)
    s.to_hdf(os.path.join(destDir, 'syntheticData.h5'), key = str(x))

for x in [3, 7, 8, 9, 10]:
    print(x)
    epoch = []
    for a in range(1, 401):
        eeg = []
        for b in range(3840):
            eeg.append(float(-0.5))
        eeg = numpy.array(eeg)
        epoch.append([x, 'TestData', eeg, 'G'])
    s = pd.Series(epoch)
    s.to_hdf(os.path.join(destDir, 'syntheticData.h5'), key = str(x))

# %%

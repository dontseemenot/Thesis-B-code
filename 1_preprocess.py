# %%
import os
import mne
from mne.filter import filter_data
import numpy as np
from numpy import testing
import matplotlib.pyplot as plt
import h5py
import csv
from datetime import datetime
import pandas as pd
import tables
from scipy import signal

downsample = True

rawDir = 'F:\Thesis B insomnia data\Insomnia data\Data_Study3\Berlin PSG EDF files'
stageDir = 'F:\\Thesis B insomnia data\\Insomnia data\\Data_Study3\\Berlin PSG annotation files'
destDir = 'F:\\Berlin data formatted'

def customFilter(y, fs, downsample):
    # 50Hz notch filter
    y2 = mne.filter.notch_filter(y, fs, 50)
    # Passband filter 0.01-50Hz. By default, 4th order butterworth filter is used
    y2 = mne.filter.filter_data(y2, fs, 0.01, 50)
    # Resample data (if needed) from 512 to 128hz
    
    if downsample:
        y2 = mne.filter.resample(y2, up = 1, down = 4)
        fs = 128
    '''
    fft = np.fft.rfft(y)
    absfft = np.abs(fft)
    power = np.square(absfft)
    frequency = np.linspace(0, fs/2, len(power))

    
    fft2 = np.fft.rfft(y2)
    absfft2 = np.abs(fft2)
    power2 = np.square(absfft2)
    frequency2 = np.linspace(0, fs/2, len(power2))
    '''
    return y2, fs

def trimStart(rawData, stageTimestamp, rawTimestamp, dataPointsInEpoch, fs):
    diff = int((stageTimestamp - rawTimestamp).total_seconds())
    print("Stage: ", stageTimestamp, "| Raw: ", rawTimestamp, " | Stage - raw = ", diff)
    offset = 0
    assert(abs(diff) < 30)
    if diff < 0:
        # Sleep stage annotated before raw data recorded. Discard raw data before 2nd sleep stage annotation and start from 2nd sleep stage.
        diffPoints = dataPointsInEpoch - abs(diff * fs)
        y = rawData[diffPoints: -1]
        offset = 1
        print("Start from 2nd sleep stage, discard prev data. 2nd sleep stage: ", lines[7 + offset])
    elif diff > 0:
        # Raw data started recording before sleep stage annotated. Discard raw data before first sleep stage annotation.
        diffPoints = abs(diff * fs)
        y = rawData[diffPoints: -1]
        print("Start from 1st sleep stage, discard prev data. 1st sleep stage: ", lines[7])
    else:
        # Sleep stage annotated at same time data recorded. Continue. 
        diffPoints = 0
        y = rawData
        print("Same start times, continue")
    return y, offset
    
# Traverse through 67 patient files
epochData = []
# Normalization bounds
lower = -1
upper = 1

df = pd.DataFrame(columns = ['pID', 'Sleep stage', 'Epoch data', 'Patient class'])   # Empty dataframe
df['Sleep stage'] = df['Sleep stage'].astype('category')
df['Patient class'] = df['Patient class'].astype('category')
for rawName, stageName, p in zip(os.listdir(rawDir), os.listdir(stageDir), range(67)):
    print(rawName)
    pID = int(rawName[3:5])
    pClass = rawName[-5]

    # Extract raw data
    raw = mne.io.read_raw_edf(os.path.join(rawDir, rawName))
    fs = int(raw.info['sfreq'])
    assert(raw['C4:A1'][1][512] == 1)   # check if original sampling freq is really 512hz
    dataPointsInEpoch = fs * 30
    rawData = raw['C4:A1'][0][0]  # raw data only
    
    # timeData = raw['C4:A1'][1]
    rawTimestamp = raw.info['meas_date']
    rawTimestamp = rawTimestamp.replace(tzinfo = None)  # Remove timezone info
    
    # Extract sleep stage annotation
    d = [filename for filename in os.listdir(os.path.join(stageDir, stageName)) if filename.startswith('Schlafprofil')]
    assert(len(d) == 1)
    with open(os.path.join(stageDir, stageName,d[0])) as f:
        lines = f.readlines()
    f.close()

    # Check if sleep stage annotation is consistent across all patients
    assert(lines[4] == 'Events list: Stadium 4,Stadium 3,Stadium 2,Stadium 1,Rem,Wach,Bewegung\n')
    assert(lines[5] == 'Rate: 30 s\n')

    numEpochs = 0
    stageTimestamp = lines[1][-20:-1]
    stageTimestamp = datetime.strptime(stageTimestamp, '%d.%m.%Y %H:%M:%S')


    
    # Power spectrum
    y, offset = trimStart(rawData, stageTimestamp, rawTimestamp, dataPointsInEpoch, fs)
    offset += 7

    y2, fs = customFilter(y, fs, downsample)

    dataPointsInEpoch = fs * 30    # If downsampled, data points will be reduced
    epochData = []
    over = 0

    # Remove >250uv epochs
    for b in range(offset, len(lines)):
    # for b in range(offset, offset + 2):
        stage = lines[b].split('; ', 1)[1].split('\n', 1)[0]
        start = (b - offset) * dataPointsInEpoch
        end = (b - offset + 1) * dataPointsInEpoch
        if end >= len(y2):
            print("Last epoch cut off! Breaking out of loop")
            break
        amplitudeData = y2[start: end]
        artefact = False
        for i in amplitudeData: # can use list comprehension instead in the future
            # Remove > 250uv artefacts
            if abs(i) > 0.000250:
                #print('artefact at {} for {}'.format(i, b))
                artefact = True
                break
        if artefact == False:
            #amplitudeDataNormalized = (amplitudeData[:] - amplitudeData[:].mean()) / (amplitudeData.std(ddof = 0))
            #amplitudeDataNormalized = amplitudeData - np.mean(amplitudeData)
            epochData.append([pID, stage, amplitudeData, pClass])
    
    # Remove DC offset (zero mean)
    dcOffset = np.mean([e for epoch in epochData for e in epoch[2]])
    for i in range(len(epochData)):
        epochData[i][2] -= dcOffset
        epochData[i][2] *= 4000     # shift data from [-250u, 250u] to [-1, 1]
        #epochData[i][2] = (epochData[i][2] - min(epochData[i][2])) / (max(epochData[i][2]) - min(epochData[i][2]))

    s = pd.Series(epochData)
    s.to_hdf(os.path.join(destDir, 'allDataNormDown2.h5'), key = str(pID))

print("Done")


 # %%

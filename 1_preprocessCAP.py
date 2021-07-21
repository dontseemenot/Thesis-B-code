# %%
import os
import mne
from mne.filter import filter_data, create_filter
import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
from datetime import datetime, date
import pandas as pd
import tables
from scipy import signal
import re

def trimStart(rawData, stageTimestamp, rawTimestamp, fs):

    a = stageTimestamp.time()
    b =  rawTimestamp.time()
    diff = int((datetime.combine(date.today(), a) - datetime.combine(date.today(), b)).total_seconds()) #  For some reason, the years in the edf and corresponding txt files can be different, so just ignore the date part. But we need to include a date in order to calculate the time difference in seconds, so use today's date.
    print("Stage: ", a, "| Raw: ", b, " | Stage - raw = ", diff)

    assert(diff >= 0)
    diffPoints = diff * fs
    y = rawData[diffPoints: None]
    return y, diff

def customFilter(y, fs):
    y2 = y
    # 50Hz notch filter
    #y2 = mne.filter.notch_filter(y, fs, 50)
    # Passband filter 0.01-50Hz.
    y2 = filter_data(y, sfreq = fs, l_freq = 0.5, h_freq = 40, l_trans_bandwidth = 0.5, h_trans_bandwidth = 0.5, method = 'fir', fir_window = 'hamming')


    # Downsample
    if fs < 128:
        y2 = mne.filter.resample(y2, up = 128/fs, down = 1)
    else:
        y2 = mne.filter.resample(y2, up = 1, down = fs/128)
    fs_new = 128
    '''
    fft = np.fft.rfft(y)
    absfft = np.abs(fft)
    power = np.square(absfft)
    frequency = np.linspace(0, fs/2, len(power))
    plt.plot(frequency, power)
    plt.show()

    fft2 = np.fft.rfft(y2)
    absfft2 = np.abs(fft2)
    power2 = np.square(absfft2)
    frequency2 = np.linspace(0, fs_new/2, len(power2))
    plt.plot(frequency2, power2)
    plt.show()
    '''
    return y2, fs_new
# %%
rawDir = 'F:\\CAP Sleep Database'
destDir = 'F:\\Sleep data formatted'
rawFiles = ['ins1.edf', 'ins2.edf', 'ins3.edf', 'ins4.edf', 'ins5.edf', 'ins6.edf', 'ins7.edf', 'ins8.edf', 'ins9.edf', 'n1.edf', 'n2.edf', 'n3.edf', 'n4.edf', 'n5.edf', 'n12.edf', 'n14.edf', 'n15.edf', 'n16.edf']
stageFiles = ['ins1.txt', 'ins2.txt', 'ins3.txt', 'ins4.txt', 'ins5.txt', 'ins6.txt', 'ins7.txt', 'ins8.txt', 'ins9.txt', 'n1.txt', 'n2.txt', 'n3.txt', 'n4.txt', 'n5.txt', 'n12.txt', 'n14.txt', 'n15.txt', 'n16.txt']

#stageFiles = ['n16.txt']
#rawFiles = ['n16.edf']
df = pd.DataFrame(columns = ['pID', 'Sleep stage', 'Epoch data', 'Patient class'])   # Empty dataframe
df['Sleep stage'] = df['Sleep stage'].astype('category')
df['Patient class'] = df['Patient class'].astype('category')

for rawF, stageF in  zip(rawFiles, stageFiles):

    ch = 'C4-A1'
    ampMult = 1
    lineOffset = 0
    # Someone decided to annotate the same channel differently across recordings
    if rawF in ['n13.edf', 'n14.edf', 'n15.edf']:
        ch = 'C4A1'
        ampMult = 1/1000000   # also they didn't convert to uV
    
    if rawF[0:3] == 'ins':
        pID = int(rawF[3])
        pClass = 'I'
    elif rawF[0] == 'n':
        pID = int(re.search('[0-9]{1,2}', rawF).group()) + 10
        pClass = 'G'
        print(pID)
    else:
        print('error! invalid file name')
        break
    raw = mne.io.read_raw_edf(os.path.join(rawDir, rawF))
    fs = int(raw.info['sfreq'])
    try:
        assert(raw[ch][1][fs] == 1)   # check fs
    except Exception as v:
        print(rawF + "invalid stated frequency!")
        exit()
    rawData = raw[ch][0][0] * ampMult  # raw data only

    rawTimestamp = raw.info['meas_date']
    rawTimestamp = rawTimestamp.replace(tzinfo = None)  # Remove timezone info

    with open(os.path.join(rawDir, stageF)) as f:
        stageLines = f.readlines()
    f.close()

    if rawF in ['n16.edf']: # inconsistent formatting
        assert(stageLines[20] == 'Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration [s]\tLocation\n')
    else:
        assert(stageLines[21] == 'Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n' or
        stageLines[21] == 'Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]' or
        stageLines[21] == 'Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n')
    
    # Extract starting timestamp from sleep stage annotation .txt file
    stageTime = re.search('\d{2}:\d{2}:\d{2}', stageLines[22]).group()
    stageDate = re.search('\d{2}/\d{2}/\d{4}', stageLines[3]).group()
    stageTimestamp = datetime.strptime(stageDate + ' ' + stageTime, '%d/%m/%Y %H:%M:%S')
    print(stageTimestamp)

    # Calculate starting point in Raw eeg data
    y, diff = trimStart(rawData, stageTimestamp, rawTimestamp, fs)
    y2, fs = customFilter(y, fs)    # returns new frequency of 128
    dataPointsInEpoch = fs * 30

    i = 0
    epochData = []
    
    for line in stageLines[22: None]:
        stage = re.search(r'SLEEP-(REM|S0|S1|S2|S3|S4)', line)
        # Ignore lines with MCAP events
        if stage is None:
            pass
        else:
            stage = stage.group()
            #print(stage)
            
            start = i*dataPointsInEpoch
            end = (i + 1) * dataPointsInEpoch
            amplitudeData = y2[start: end]
            
            # If last annotated epoch was cut off early (epoch duration < 30s), cut off last epoch
            if end >= len(y2):
                print("Last epoch cut off! Breaking out of loop")
                break
            i += 1
            # Remove epochs with artefacts
            artefact = False
            for j in amplitudeData:
                if abs(j) > 0.000250:
                    artefact = True
                    print("artefact at ", i)
                    break
            if artefact == False:
                epochData.append([pID, stage, amplitudeData, pClass])

            # Remove DC offset
    ###
    dcOffset = np.mean([e for epoch in epochData for e in epoch[2]])
    # Remove offset and normalize from [-250, 250uV] to [-1, 1]
    ###
    for i in range(len(epochData)):
        epochData[i][2] -= dcOffset
        epochData[i][2] *= 4000
    
    
    s = pd.Series(epochData)
    s.to_hdf(os.path.join(destDir, 'allCAP.h5'), key = str(pID))


        #time = datetime.strptime(match.group(), '%Y-%m-%d').time()
        
    
    



# %%

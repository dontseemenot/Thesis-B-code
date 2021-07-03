# %%
import os
import mne
import numpy as np 
import matplotlib.pyplot as plt
import h5py
import csv
from datetime import datetime
import pandas as pd
import tables




# %%
rawDir = 'F:\Thesis B insomnia data\Insomnia data\Data_Study3\Berlin PSG EDF files'
stageDir = 'F:\\Thesis B insomnia data\\Insomnia data\\Data_Study3\\Berlin PSG annotation files'
destDir = 'F:\\Berlin data formatted'
# %%
dataPointsInEpoch = 512 * 30

stageDict = {
    'Wach': 0,      # Wake
    'Stadium 1': 1,
    'Stadium 2': 2,
    'Stadium 3': 3,
    'Stadium 4': 4,
    'Rem': 5,
    'Bewegung': 6,  # Movement
    'A': 7          # ???
}



# Traverse through 67 patient files
epochData = []
df = pd.DataFrame(columns = ['pID', 'Sleep stage', 'Epoch data', 'Patient class'])   # Empty dataframe
df['Sleep stage'] = df['Sleep stage'].astype('category')
df['Patient class'] = df['Patient class'].astype('category')
for rawName, stageName, p in zip(os.listdir(rawDir), os.listdir(stageDir), range(67)):
    print(rawName)
    pID = int(rawName[3:5])
    pClass = rawName[-5]
    # Classify patient as Healthy (G) or Insomniac (I)
    '''
    if rawName[-5] == 'G':
        pClass = 0
    elif rawName[-5] == 'I':
        pClass = 1
    else:
        pClass = 2
    '''
    # Extract raw data
    raw = mne.io.read_raw_edf(os.path.join(rawDir, rawName))
    samplingFreq = int(raw.info['sfreq'])
    rawData = raw['C4:A1'][0][0]  # raw data only

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
    stageData = []
    stageTimestamp = lines[1][-20:-1]
    stageTimestamp = datetime.strptime(stageTimestamp, '%d.%m.%Y %H:%M:%S')

# Make sure the first sleep stage annotation corresponds to the epoch with correct timestamp
    diff = int((stageTimestamp - rawTimestamp).total_seconds())
    print("Stage: ", stageTimestamp, "| Raw: ", rawTimestamp, " | Stage - raw = ", diff)
    offset = 0
    if abs(diff) >= 30:
        print('Error! diff >= 30')
        break
    if diff < 0:
        # Sleep stage annotated before raw data recorded. Discard raw data before 2nd sleep stage annotation and start from 2nd sleep stage.
        diffPoints = abs(diff * 512)
        y = rawData[diffPoints: -1]
        offset = 1
        print("Start from 2nd sleep stage, discard prev data. 2nd sleep stage: ", lines[7 + offset])
    elif diff > 0:
        # Raw data started recording before sleep stage annotated. Discard raw data before first sleep stage annotation.
        diffPoints = abs(diff * 512)
        y = rawData[diffPoints: -1]
        print("Start from 1st sleep stage, discard prev data. 1st sleep stage: ", lines[7])
    else:
        # Sleep stage annotated at same time data recorded. Continue. 
        diffPoints = 0
        y = rawData
        print("Same start times, continue")
    assert(y[0] == rawData[diffPoints])
    
    offset += 7
    epochData = []
    for b in range(offset, len(lines)):
        stage = lines[b].split('; ', 1)[1].split('\n', 1)[0]
        start = (b - offset) * dataPointsInEpoch
        end = (b - offset + 1) * dataPointsInEpoch
        amplitudeData = y[start: end]
        # There were more sleep stage annotations than available raw data
        if len(amplitudeData) != dataPointsInEpoch:
            print("Last epoch cut off! Breaking out of loop")
            break
        numEpochs += 1
        epochData.append([pID, stage, amplitudeData, pClass])
    s = pd.Series(epochData)
    s.to_hdf(os.path.join(destDir, 'allData.h5'), key = str(pID))



print("Done")
# %%

# %%
import os
import mne
from mne.filter import filter_data, create_filter
from mne.time_frequency import tfr_array_multitaper
import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
from datetime import datetime, date, timedelta
import pandas as pd
import tables
from scipy import signal
import re
from preprocess_parameters import *
import gc
# %%
def trimStartBerlin(rawData, startTime, rawTimestamp, dataPointsInEpoch, fs, lines):
    diff = int((startTime - rawTimestamp).total_seconds())
    # print("Stage: ", startTime, "| Raw: ", rawTimestamp, " | Stage - raw = ", diff)
    offset = 0
    assert(abs(diff) < 30)
    if diff < 0:
        # Sleep stage annotated before raw data recorded. Discard raw data before 2nd sleep stage annotation and start from 2nd sleep stage.
        diffPoints = dataPointsInEpoch - abs(diff * fs)
        y = rawData[diffPoints: -1]
        offset = 1
        # print("Start from 2nd sleep stage, discard prev data. 2nd sleep stage: ", lines[7 + offset])
    elif diff > 0:
        # Raw data started recording before sleep stage annotated. Discard raw data before first sleep stage annotation.
        diffPoints = abs(diff * fs)
        y = rawData[diffPoints: -1]
        # print("Start from 1st sleep stage, discard prev data. 1st sleep stage: ", lines[7])
    else:
        # Sleep stage annotated at same time data recorded. Continue. 
        diffPoints = 0
        y = rawData
        # print("Same start times, continue")
    return y, offset
    

def trimStartCAP(rawData, startTime, rawTimestamp, fs):

    a = startTime.time()
    b =  rawTimestamp.time()
    diff = int((datetime.combine(date.today(), a) - datetime.combine(date.today(), b)).total_seconds()) #  For some reason, the years in the edf and corresponding txt files can be different, so just ignore the date part. But we need to include a date in order to calculate the time difference in seconds, so use today's date.
    #print("Stage: ", a, "| Raw: ", b, " | Stage - raw = ", diff) in seconds

    assert(diff >= 0)
    diffPoints = diff * fs
    y = rawData[diffPoints: None]
    offset = 0
    return y, offset

def customFilter(y, fs):
    y2 = y
    # Downsample
    #print(f'Original fs: {fs}, length: {len(y2)}')
    if fs < 128:
        y2 = mne.filter.resample(y2, up = 128/fs, down = 1, verbose = False)
    elif fs > 128:
        y2 = mne.filter.resample(y2, up = 1, down = fs/128, verbose = False)
    fs = 128
    # 50Hz notch filter

    #print(f'Intermediate fs: {fs}, length: {len(y2)}')
    # y2 = filter_data(y2, sfreq = fs, l_freq = None, h_freq = 40, l_trans_bandwidth = None, h_trans_bandwidth = 10, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)

    # Passband 0-5.40Hz filter
    # Guidelines for EEG signal filtering from https://mne.tools/dev/auto_tutorials/preprocessing/25_background_filtering.html
    # FIR
    # High-pass transition band: 0.5Hz
    # Low-pass transition band: 2Hz
    # Zero-phase
    # Hamming window

    y2 = filter_data(y2, sfreq = fs, l_freq = 0.5, h_freq = 40, l_trans_bandwidth = 0.5, h_trans_bandwidth = 2, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
    #print(f'New fs: {fs}, length: {len(y2)}')

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
    return y2, fs

#5s window overlapping
# Eg: [abcdef, ghijkl] --> [abcdef, bcdefg, cdefgh, defghi, efghij, fghijk, ghijkl]
def overlap(epochData1D, dataPointsInEpoch):
    extraWindows = 5    # 25s overlapping windows between 30s epochs
    offset = int(dataPointsInEpoch / (extraWindows + 1))
    epochData1D2 = []
    i = 0
    j = 0
    curStage = epochData1D[0][0]
    epochData1D2.append(epochData1D[0])
    for i in range(1, len(epochData1D)):
    #for i in range(1, 4):  # debugging
        if epochData1D[i][0] == curStage:
            # Do overlap if two consecutive epochs are the same stage
            epochCombined = np.array([*epochData1D[i - 1][1], *epochData1D[i][1]])
            assert(len(epochCombined) == dataPointsInEpoch*2)
            # Do overlap
            for j in range(1, extraWindows + 1):    # j = 1 to 5
                epochData1D2.append([epochData1D[i][0], epochCombined[j*offset: j*offset + dataPointsInEpoch]])
            curStage = epochData1D[i][0]
            assert(np.all(
                epochCombined[(2*offset):(2*offset + dataPointsInEpoch)] ==
                [*epochData1D2[-4][1]]))  # Sanity check
        epochData1D2.append(epochData1D[i])
        assert(np.all(epochData1D[i][1] == epochData1D2[-1][1]))    # Sanity check
        curStage = epochData1D[i][0]
            
        
    #print(f'len1 {len(epochData1D)}, len2 {len(epochData1D2)}')
    return epochData1D2

# def removeArtefacts(epochData1D):
#     epochData1D2 = [epoch for epoch in epochData1D if all(abs(point) <= 250e-6 for point in epoch[1])]

#     #print('artefacts removed ', len(epochData1D) - len(epochData1D2))

#     return epochData1D2

# Given an array a and b, get indices of artefacts to remove in b, then apply to a and return a
def removeArtefacts2(a):
    #assert(len(a) == len(b))
    #idxs = np.zeros(len(a))
    #print(f"len(a) = {len(a)}")
    c = []
    for epoch_a in a:
        if all(abs(point) < 250e-6 for point in epoch_a[1]):
            c.append(epoch_a)

    #print(f"new len: {len(c)}")
    return c

def removeDCOffset(epochData1D):
    localMeans = np.mean([data for epoch in epochData1D for data in epoch[1]])    # mean for each epoch
    dcOffset = np.mean(localMeans)  # global mean
    #print('dcoffset ', dcOffset)
    epochData1D2 = []
    for epoch in epochData1D:
        epochData1D2.append([epoch[0], epoch[1] - dcOffset])
    return epochData1D2

def scale(epochData1D):
    scaleFactor = 4000
    epochData1D2 = []
    for epoch in epochData1D:
        epochData1D2.append([epoch[0], epoch[1]*scaleFactor])
    return epochData1D2

def countSleepStages(sleep_stages_list):
    num_W = 0
    num_S1 = 0
    num_S2 = 0
    num_S3 = 0
    num_S4 = 0
    num_R = 0
    num_other = 0
    num_all = 0
    for s in sleep_stages_list:
        num_all += 1
        if s == 'W':
            num_W += 1
        elif s == 'S1':
            num_S1 += 1
        elif s == 'S2':
            num_S2 += 1
        elif s == 'S3':
            num_S3 += 1
        elif s == 'S4':
            num_S4 += 1
        elif s == 'R':
            num_R += 1
        else:
            num_other += 1
    #print(f"Stages: W {num_W} S1 {num_S1} S2 {num_S2} S3 {num_S3} S4 {num_S4} R {num_R}")
    return ({'W': num_W, 'S1': num_S1, 'S2': num_S2, 'S3': num_S3, 'S4': num_S4, 'R': num_R, 'Other': num_other, 'All': num_all})

def calculate_epochs_between_times(startTime, endTime):
    a = datetime.strptime(startTime, '%H:%M:%S')
    b = datetime.strptime(endTime, '%H:%M:%S')
    seconds_passed = (b - a).total_seconds()
    if seconds_passed < 0:
        seconds_passed += 86400 # seconds in a day
    numEpochs = int(seconds_passed) / int(30)
    return int(numEpochs)

def customTFR(y, sfreq):
    gc.collect()
    f_min = 0.1
    f_max = 50.0
    freqs = np.linspace(f_min, f_max, num = 224, endpoint = True)

    n_cycles = freqs / 2
    time_bandwidth = 3.0
    print(f"y shape: {y.shape}")
    y = y.reshape((1, 1, y.shape[0]))
    print(f"y shape: {y.shape}")
    z = tfr_array_multitaper(y, sfreq = sfreq, freqs = freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, output = 'power')
    x = np.arange(1, 3841, 1)
    plt.pcolormesh(x, freqs, z, shading = 'flat', cmap = 'gray')
    plt.title(f"BW = {time_bandwidth}")
    plt.show()
    plt.plot(z)
    plt.show()
    return z
# rawF: File path to raw .edf file
# stageF: File path to annotation .txt file
# pID: patient ID
# ch: Channel to extract raw data from
# amp: Constant to multiply signal with (ensure consistensy between files)
# dataset_name: CAP or Berlin
def annotateData(rawF, stageF, pID, ch, amp, dataset_name):

    # Get patient ID and class
    # Channel and units were represented differently in these files
    if dataset_name == 'CAP':
        if pID[0] == 'n':
            pClass = 'G'
        elif pID[0:3] == 'ins':
            pClass = 'I'
        sleepDict = {'SLEEP-S0': 'W', 'SLEEP-S1': 'S1', 'SLEEP-S2': 'S2', 'SLEEP-S3': 'S3', 'SLEEP-S4': 'S4', 'SLEEP-REM': 'R', 'Other': 'Other'}
        regexStage = r'SLEEP-(REM|S0|S1|S2|S3|S4)'
        regexTime = '\d{2}:\d{2}:\d{2}'
    elif dataset_name == 'Berlin':
        if pID[6] == 'G':
            pClass = 'G'
        elif pID[6] == 'I':
            pClass = 'I'
        sleepDict = {'Wach': 'W', 'Stadium 1': 'S1', 'Stadium 2': 'S2', 'Stadium 3': 'S3', 'Stadium 4': 'S4', 'Rem': 'R', 'A': 'Other', 'Bewegung': 'Other'}
        regexStage = r'Wach|Stadium 1|Stadium 2|Stadium 3|Stadium 4|Rem|A|Bewegung'
        regexTime = '\d{2}:\d{2}:\d{2}'

    

    # Read raw data
    raw = mne.io.read_raw_edf(rawF, verbose = False)
    original_fs = int(raw.info['sfreq'])
    assert(raw[ch][1][original_fs] == 1)   # check signal if listed frequency is true

    rawData = raw[ch][0][0] * amp   # For CAP data, units may not be in uV so need to correct this
    print(f"length rawdata {len(rawData)}")
    rawTimestamp = raw.info['meas_date']
    rawTimestamp = rawTimestamp.replace(tzinfo = None)  # Remove timezone info

    # Frequency filter
    y, fs = customFilter(rawData, original_fs) # Low pass 50Hz filter and downsample to 128Hz
    dataPointsInEpoch = fs * 30
    epochData1D = []
    epochData2D = []

    # Read sleep stage annotation file
    with open(stageF) as f:
        stageLines = f.readlines()
    if dataset_name == 'Berlin':
        # Check if sleep stage annotation is consistent across all patients
        assert(stageLines[4] == 'Events list: Stadium 4,Stadium 3,Stadium 2,Stadium 1,Rem,Wach,Bewegung\n')
        assert(stageLines[5] == 'Rate: 30 s\n')
        startTime = stageLines[1][-20:-1]
        startTime = datetime.strptime(startTime, '%d.%m.%Y %H:%M:%S')
        y, offset = trimStartBerlin(y, startTime, rawTimestamp, dataPointsInEpoch, fs, stageLines)    # Ignore data at the beginning which do not contain annotations
        startIndex = 7  # Index of line with first annotation
        i = 0

    elif dataset_name == 'CAP':
        # Extract starting timestamp from sleep stage annotation .txt file
        stageTime = re.search('\d{2}:\d{2}:\d{2}', stageLines[22]).group()
        stageDate = re.search('\d{2}/\d{2}/\d{4}', stageLines[3]).group()
        startRecordTime = datetime.strptime(stageDate + ' ' + stageTime, '%d/%m/%Y %H:%M:%S')
        y, offset = trimStartCAP(y, startRecordTime, rawTimestamp, fs) # Ignore data at the beginning which do not contain annotations
        startIndex = 22
        i = 0

    # Get the TFR spectrogram 
    #z = customTFR(y, fs)
    z = y
    # Append the first stage, assuming it is properly labelled
    stage = re.search(regexStage, stageLines[i + startIndex + offset]).group()
    startTime = re.search(regexTime, stageLines[i + startIndex + offset]).group()
    start = i*dataPointsInEpoch
    end = (i + 1) * dataPointsInEpoch
    amplitudeData1D = y[start: end]
    amplitudeData2D = z[start: end]
    epochData1D.append([sleepDict[stage], amplitudeData1D])
    epochData2D.append([sleepDict[stage], amplitudeData2D])
    prevTime = startTime
    i += 1
    # Should be the same for both CAP and Berlin
    #print(regexStage)
    for line in stageLines[(i + startIndex + offset): None]:
        stage = re.search(regexStage, line)
        if stage is None:
            pass    # Ignore lines with MCAP events or other labels
        else:
            stage = stage.group()
            curTime = re.search(regexTime, line).group()
            unlabelled_epochs = calculate_epochs_between_times(prevTime, curTime) - 1
            # Account for unlabelled epochs
            if unlabelled_epochs != 0:
                #print(f'Unlabelled epochs = {unlabelled_epochs} for line {line}')
                for j in range(unlabelled_epochs):
                    start = i*dataPointsInEpoch
                    end = (i + 1) * dataPointsInEpoch
                    amplitudeData1D = y[start: end]
                    amplitudeData2D = z[start: end]
                    prev_sleepDict = epochData1D[-1][0]   # Use last known epoch sleep label for unlabelled epoch
                    #print(f'Appending {prev_sleepDict}')
                    epochData1D.append([prev_sleepDict, amplitudeData1D])
                    epochData2D.append([prev_sleepDict, amplitudeData2D])
                    i += 1
            
            
            start = i*dataPointsInEpoch
            end = (i + 1) * dataPointsInEpoch
            amplitudeData1D = y[start: end]
            amplitudeData2D = z[start: end]
            
            # If last annotated epoch was cut off early (epoch duration < 30s), cut off last epoch
            if end > len(y):
                #print(f'len y2 {len(y2)} end {end}')
                #print("Last epoch cut off! Breaking out of loop")
                break
            i += 1
            epochData1D.append([sleepDict[stage], amplitudeData1D])
            epochData2D.append([sleepDict[stage], amplitudeData2D])
            # Get timestamp of current epoch
            endTime = re.search(regexTime, line).group() 
            prevTime = curTime
    #print(endTime)
    # Add 30s to endtime to denote when epoch recording stopped
    # endTime = datetime.strptime(endTime, '%H:%M:%S')
    # delta = timedelta(seconds = 30)
    # endTime += delta
    # endTime = endTime.strftime('%H:%M:%S')
    numEpochs = calculate_epochs_between_times(startTime, endTime) + 1
    # print(numEpochs, i)
    # print(f'starttime {startTime}, endtime {endTime}')
    print(f'i {i}, numEpochs {numEpochs}, start {startIndex}, offset {offset}')
    assert(numEpochs == i)

    #print(endTime)
    return epochData1D, epochData2D, dataPointsInEpoch, pID, pClass, startTime, endTime, numEpochs, original_fs, fs



print(f"Dataset: {dataset} Overlap: {overlapBool}")


if dataset == 'Berlin':
    pIDs = [
        'IM_01 I', 'IM_02 I', 'IM_03 G', 'IM_04 I', 'IM_05 I', 'IM_06 I', 'IM_07 G', 'IM_08 G', 'IM_09 G', 'IM_10 G', 'IM_11 G', 'IM_12 G', 'IM_15 I', 'IM_16 I', 'IM_17 I', 'IM_18 I', 'IM_19 I', 'IM_20 I', 'IM_21 I', 'IM_22 G', 'IM_24 G', 'IM_26 I', 'IM_27 I', 'IM_28 G', 'IM_29 G', 'IM_30 G', 'IM_31 G', 'IM_32 G', 'IM_33 G', 'IM_34 G', 'IM_35 G', 'IM_36 G', 'IM_38 G', 'IM_39 G', 'IM_40 G', 'IM_41 I', 'IM_42 I', 'IM_43 I', 'IM_44 G', 'IM_45 G', 'IM_46 G', 'IM_47 G', 'IM_48 G', 'IM_49 G', 'IM_50 G', 'IM_51 G', 'IM_52 I', 'IM_53 I', 'IM_54 I', 'IM_55 I', 'IM_56 I', 'IM_57 I', 'IM_59 G', 'IM_60 I', 'IM_61 G', 'IM_62 I', 'IM_63 I', 'IM_64 I', 'IM_65 G', 'IM_66 I', 'IM_68 I', 'IM_69 I', 'IM_70 I', 'IM_71 I', 'IM_73 I', 'IM_74 I', 'IM_75 I'
    ]
    rawDir = rawDirBerlin
    rawFiles = [rawDir + f'/{pID}.edf'  for pID in pIDs]
    stageDir = stageDirBerlin
    stageFiles = [filename for pID in pIDs for filename in os.listdir(os.path.join(stageDir, f'{pID}' + '/')) if filename.startswith('Schlafprofil')]
    stageFiles = [stageDir + f'/{pID}/' + s for s, pID in zip(stageFiles, pIDs)]
    chNames = ['C4:A1' for i in range(len(stageFiles))]    # In Berlin dataset, channel name is consistent
    ampMults = [1 for i in range(len(stageFiles))] # In Berlin dataset, signal unit is consistent

    pid_max_epochs = []

elif dataset == 'CAP':

    pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16']

    rawDir = rawDirCAP
    rawFiles = [
        'ins1.edf', 'ins2.edf', 'ins3.edf', 'ins4.edf', 'ins5.edf', 'ins6.edf', 'ins7.edf', 'ins8.edf', 'ins9.edf',
        'n1.edf', 'n2.edf', 'n3.edf', 'n4.edf', 'n5.edf', 'n6.edf', 'n7.edf', 'n8.edf', 'n9.edf', 'n10.edf', 'n11.edf', 'n12.edf', 'n13.edf', 'n14.edf', 'n15.edf', 'n16.edf',
    ]
    rawFiles = [rawDir + s for s in rawFiles]

    stageDir = stageDirCAP
    stageFiles = [
        'ins1.txt', 'ins2.txt', 'ins3.txt', 'ins4.txt', 'ins5.txt', 'ins6.txt', 'ins7.txt', 'ins8.txt', 'ins9.txt',
        'n1.txt', 'n2.txt', 'n3.txt', 'n4.txt', 'n5.txt', 'n6.txt', 'n7.txt', 'n8.txt', 'n9.txt', 'n10.txt', 'n11.txt', 'n12.txt', 'n13.txt', 'n14.txt', 'n15.txt', 'n16.txt'
    ]
    stageFiles = [stageDir + s for s in stageFiles]

    chNames = [
        # ins1-9
        'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1',
        # n1-16
        'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C4-A1', 'C3-A2', 'C3-A2', 'C3-A2', 'C3-A2',
        'C4-A1',  'C4-A1', 'C4-A1', 'C3A2', 'C4A1', 'C4A1', 'C4-A1']
    ampMults = [
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1e-3, 1, 1, 1, 1, 1e-6, 1e-6, 1e-6, 1
    ]
    pid_max_epochs = {'n10': 351, 'n11': 362, 'n14': 767}


#
#
pIDs = ['IM_01 I']
rawFiles = [rawDir + f'/{pID}.edf'  for pID in pIDs]
stageFiles = [filename for pID in pIDs for filename in os.listdir(os.path.join(stageDir, f'{pID}' + '/')) if filename.startswith('Schlafprofil')]
stageFiles = [stageDir + f'/{pID}/' + s for s, pID in zip(stageFiles, pIDs)]
#
metadata_list = []
# %%
for i, rawF, stageF, pID, ch, amp in zip(range(100), rawFiles, stageFiles, pIDs, chNames, ampMults):
    print(f'Preprocessing {pID}...')
    epochData1D, dataPointsInEpoch, pID, pClass, startTime, endTime, numEpochs, original_fs, fs  = annotateData(rawF, stageF, pID, ch, amp, dataset)


    # if pID in pid_max_epochs:
    #     epochData1D = epochData1D[:pid_max_epochs[pID]]
    
    if overlapBool == True:
        epochData1D = overlap(epochData1D, dataPointsInEpoch)
    

    epochData1D = removeArtefacts2(epochData1D)
        # epochData1D = removeDCOffset(epochData1D)
    (sleep_stage_count) = countSleepStages([epoch[0] for epoch in epochData1D]) # returns W, 
    metadata = (pID, pClass, startTime, endTime, original_fs, fs, sleep_stage_count['W'], sleep_stage_count['S1'], sleep_stage_count['S2'], sleep_stage_count['S3'], sleep_stage_count['S4'], sleep_stage_count['R'], sleep_stage_count['Other'], sleep_stage_count['All'])
    print(metadata)
    metadata_list.append(metadata)
    dataCols = [f'X{x}' for x in range(1, dataPointsInEpoch + 1)]
    columns = ['Sleep_Stage', *dataCols]
    data = [[str(epoch[0]), *epoch[1]] for epoch in epochData1D]

    df = pd.DataFrame(columns = columns, data = data)
# %%
    store = pd.HDFStore(dest_file_h5)
    store.put(pID, df)
    store.get_storer(pID).attrs.metadata = {'metadata': metadata}
    store.close()

df_meta_original = pd.DataFrame(columns = ['pID', 'pClass', 'Start time', 'End time', 'Original Fs', 'Fs', 'W', 'S1', 'S2', 'S3', 'S4', 'R', 'Other', 'Total'], data = metadata_list)
df_meta_original.to_csv(f'E:/HDD documents/University/Thesis/Thesis B code/data/{destName}_metadata.csv')
print('All done')

# %%

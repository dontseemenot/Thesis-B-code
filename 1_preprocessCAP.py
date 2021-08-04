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

def trimStartBerlin(rawData, stageTimestamp, rawTimestamp, dataPointsInEpoch, fs, lines):
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
    

def trimStartCAP(rawData, stageTimestamp, rawTimestamp, fs):

    a = stageTimestamp.time()
    b =  rawTimestamp.time()
    diff = int((datetime.combine(date.today(), a) - datetime.combine(date.today(), b)).total_seconds()) #  For some reason, the years in the edf and corresponding txt files can be different, so just ignore the date part. But we need to include a date in order to calculate the time difference in seconds, so use today's date.
    #print("Stage: ", a, "| Raw: ", b, " | Stage - raw = ", diff)

    assert(diff >= 0)
    diffPoints = diff * fs
    y = rawData[diffPoints: None]
    return y, diff

def customFilter(y, fs):
    y2 = y
    # 50Hz notch filter
    # Passband filter 0.01-50Hz.
    y2 = filter_data(y, sfreq = fs, l_freq = 0.5, h_freq = 40, l_trans_bandwidth = 0.5, h_trans_bandwidth = 0.5, method = 'fir', fir_window = 'hamming', verbose = False)
    # Downsample
    if fs < 128:
        y2 = mne.filter.resample(y2, up = 128/fs, down = 1, verbose = False)
    else:
        y2 = mne.filter.resample(y2, up = 1, down = fs/128, verbose = False)
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

#5s window overlapping
def overlap(epochData, dataPointsInEpoch):
    extraWindows = 5    # 25s overlapping windows between 30s epochs
    offset = int(dataPointsInEpoch / (extraWindows + 1))
    epochData2 = []
    i = 0
    j = 0
    curStage = epochData[0][0]
    epochData2.append(epochData[0])
    for i in range(1, len(epochData)):
    #for i in range(1, 4):  # debugging
        if epochData[i][0] == curStage:
            epochCombined = np.array([*epochData[i - 1][1], *epochData[i][1]])
            assert(len(epochCombined) == dataPointsInEpoch*2)
            # Do overlap
            for j in range(1, extraWindows + 1):    # j = 1 to 5
                epochData2.append([epochData[i][0], epochCombined[j*offset: j*offset + dataPointsInEpoch]])
            curStage = epochData[i][0]
            assert(np.all(
                epochCombined[(2*offset):(2*offset + dataPointsInEpoch)] ==
                [*epochData2[-4][1]]))  # Sanity check
        epochData2.append(epochData[i])
        assert(np.all(epochData[i][1] == epochData2[-1][1]))    # Sanity check
        curStage = epochData[i][0]
            
        
    print(f'len1 {len(epochData)}, len2 {len(epochData2)}')
    return epochData2

def removeArtefacts(epochData):
    epochData2 = [epoch for epoch in epochData if all(abs(point) <= 250e-6 for point in epoch[1])]

    print('artefacts removed ', len(epochData) - len(epochData2))

    return epochData2

def removeDCOffset(epochData):
    localMeans = np.mean([data for epoch in epochData for data in epoch[1]])    # mean for each epoch
    dcOffset = np.mean(localMeans)  # global mean
    print('dcoffset ', dcOffset)
    epochData2 = []
    for epoch in epochData:
        epochData2.append([epoch[0], epoch[1] - dcOffset])
    return epochData2

def scale(epochData):
    scaleFactor = 4000
    epochData2 = []
    for epoch in epochData:
        epochData2.append([epoch[0], epoch[1]*scaleFactor])
    return epochData2

def countSleepStages(sleep_stages_list):
    num_W = 0
    num_S1 = 0
    num_S2 = 0
    num_S3 = 0
    num_S4 = 0
    num_R = 0
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
    
    return ({'W': num_W, 'S1': num_S1, 'S2': num_S2, 'S3': num_S3, 'S4': num_S4, 'R': num_R, 'All': num_all})

def annotateDataCAP(rawF, stageF):
    
    ch = 'C4-A1'
    ampMult = 1
    lineOffset = 0
    # Channel and units were represented differently in these files
    if rawF in ['n13.edf', 'n14.edf', 'n15.edf']:
        ch = 'C4A1'
        ampMult = 1/1000000   # also they didn't convert to uV
    
    if rawF[0:3] == 'ins':
        #pID = int(rawF[3])
        pID = rawF[0:4]
        pClass = 'I'
    elif rawF[0] == 'n':
        #pID = int(re.search('[0-9]{1,2}', rawF).group()) + 10
        pID = re.search('n[0-9]{1,2}', rawF).group()
        pClass = 'G'

    else:
        print("Invalid file name")
        exit()
    print(f'pID: {pID}')
    raw = mne.io.read_raw_edf(os.path.join(rawDir, rawF), verbose = False)
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

    # Check if sleep stage column is annotated correctly
    if rawF in ['n16.edf']:
        assert(stageLines[20] == 'Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration [s]\tLocation\n')
    else:
        assert(stageLines[21] == 'Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n' or
        stageLines[21] == 'Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]' or
        stageLines[21] == 'Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n')
    
    # Extract starting timestamp from sleep stage annotation .txt file
    stageTime = re.search('\d{2}:\d{2}:\d{2}', stageLines[22]).group()
    stageDate = re.search('\d{2}/\d{2}/\d{4}', stageLines[3]).group()
    stageTimestamp = datetime.strptime(stageDate + ' ' + stageTime, '%d/%m/%Y %H:%M:%S')
    # print(stageTimestamp)

    # Calculate starting point in Raw eeg data
    y, diff = trimStartCAP(rawData, stageTimestamp, rawTimestamp, fs)
    y2, fs = customFilter(y, fs)    # returns new frequency of 128
    dataPointsInEpoch = fs * 30

    sleepDict = {'SLEEP-S0': 'W', 'SLEEP-S1': 'S1', 'SLEEP-S2': 'S2', 'SLEEP-S3': 'S3', 'SLEEP-S4': 'S4', 'SLEEP-REM': 'R'}
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
                # print("Last epoch cut off! Breaking out of loop")
                break
            i += 1
            epochData.append([sleepDict[stage], amplitudeData])
        
    return epochData, dataPointsInEpoch, pID, pClass

def annotateDataBerlin(rawDir, stageDir, rawName, stageName):
    
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
    y, offset = trimStartBerlin(rawData, stageTimestamp, rawTimestamp, dataPointsInEpoch, fs, lines)
    offset += 7

    y2, fs = customFilter(y, fs)    # returns new frequency of 128
    dataPointsInEpoch = fs * 30    # If downsampled, data points will be reduced
    epochData = []
    over = 0

    sleepDict = {'Wach': 'W', 'Stadium 1': 'S1', 'Stadium 2': 'S2', 'Stadium 3': 'S3', 'Stadium 4': 'S4', 'Rem': 'R'}

    for b in range(offset, len(lines)):
        stage = lines[b].split('; ', 1)[1].split('\n', 1)[0]
        start = (b - offset) * dataPointsInEpoch
        end = (b - offset + 1) * dataPointsInEpoch
        if end >= len(y2):
            print("Last epoch cut off! Breaking out of loop")
            break
        amplitudeData = y2[start: end]
        if sleepDict.get(stage) is not None:
            epochData.append([sleepDict[stage], amplitudeData])
        else:
            print(stage, "does not exist in sleepDict")

    return epochData, dataPointsInEpoch, str(pID), pClass
    

#Change this
dataset = "Berlin"

if dataset == 'Berlin':
    rawDir = 'F:\Thesis B insomnia data\Insomnia data\Data_Study3\Berlin PSG EDF files'
    stageDir = 'F:\\Thesis B insomnia data\\Insomnia data\\Data_Study3\\Berlin PSG annotation files'
    destDir = 'F:\\Berlin data formatted'
    dest_file_h5 = 'F:/Sleep data formatted/Berlin.h5'
    for rawName, stageName, p in zip(os.listdir(rawDir), os.listdir(stageDir), range(67)):
        epochData, dataPointsInEpoch, pID, pClass = annotateDataBerlin(rawDir, stageDir, rawName, stageName)

        # epochData = overlap(epochData, dataPointsInEpoch)
        epochData = removeDCOffset(epochData)
        # epochData = scale(epochData)
        epochData = removeArtefacts(epochData)  # We need to do this last to ensure when we scale the data later, all values lie within [-1, 1]
        (sleep_stage_count) = countSleepStages([epoch[0] for epoch in epochData]) # returns W, S1, ... R, total/all
        print(sleep_stage_count)
        dataCols = [f'X{x}' for x in range(1, dataPointsInEpoch + 1)]
        columns = ['Sleep_Stage', *dataCols]
        data = [[str(epoch[0]), *epoch[1]] for epoch in epochData]

        df = pd.DataFrame(columns = columns, data = data)
        

        store = pd.HDFStore(dest_file_h5)
        store.put(pID, df)
        metadata = (pID, pClass, (sleep_stage_count))
        store.get_storer(pID).attrs.metadata = metadata
        store.close()
    # s = pd.Series(epochData)
    # s.to_hdf(os.path.join(destDir, 'Berlin.h5'), key = str(pID))




elif dataset == 'CAP':
    rawDir = 'F:\\CAP Sleep Database'
    dest_file_h5 = 'F:/Sleep data formatted/CAP_3.h5'
    rawFiles = ['ins1.edf', 'ins2.edf', 'ins3.edf', 'ins4.edf', 'ins5.edf', 'ins6.edf', 'ins7.edf', 'ins8.edf', 'ins9.edf', 'n1.edf', 'n2.edf', 'n3.edf', 'n4.edf', 'n5.edf', 'n12.edf', 'n14.edf', 'n15.edf', 'n16.edf']
    stageFiles = ['ins1.txt', 'ins2.txt', 'ins3.txt', 'ins4.txt', 'ins5.txt', 'ins6.txt', 'ins7.txt', 'ins8.txt', 'ins9.txt', 'n1.txt', 'n2.txt', 'n3.txt', 'n4.txt', 'n5.txt', 'n12.txt', 'n14.txt', 'n15.txt', 'n16.txt']
    #stageFiles = ['n2.txt']
    #rawFiles = ['n2.edf']
    for rawF, stageF in  zip(rawFiles, stageFiles):

        epochData, dataPointsInEpoch, pID, pClass = annotateDataCAP(rawF, stageF)
        
        epochData = overlap(epochData, dataPointsInEpoch)
        epochData = removeDCOffset(epochData)
        # epochData = scale(epochData)
        epochData = removeArtefacts(epochData)  # We need to do this last to ensure when we scale the data later, all values lie within [-1, 1]
        (sleep_stage_count) = countSleepStages([epoch[0] for epoch in epochData]) # returns W, S1, ... R, total/all
        print(sleep_stage_count)
        dataCols = [f'X{x}' for x in range(1, dataPointsInEpoch + 1)]
        columns = ['Sleep_Stage', *dataCols]
        data = [[str(epoch[0]), *epoch[1]] for epoch in epochData]

        df = pd.DataFrame(columns = columns, data = data)
        

        store = pd.HDFStore(dest_file_h5)
        store.put(pID, df)
        metadata = (pID, pClass, (sleep_stage_count))
        store.get_storer(pID).attrs.metadata = metadata
        store.close()
# %%

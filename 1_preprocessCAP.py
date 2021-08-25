# %%
import os
import mne
from mne.filter import filter_data, create_filter
import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
from datetime import datetime, date, timedelta
import pandas as pd
import tables
from scipy import signal
import re

def trimStartBerlin(rawData, startTime, rawTimestamp, dataPointsInEpoch, fs, lines):
    diff = int((startTime - rawTimestamp).total_seconds())
    print("Stage: ", startTime, "| Raw: ", rawTimestamp, " | Stage - raw = ", diff)
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
    

def trimStartCAP(rawData, startTime, rawTimestamp, fs):

    a = startTime.time()
    b =  rawTimestamp.time()
    diff = int((datetime.combine(date.today(), a) - datetime.combine(date.today(), b)).total_seconds()) #  For some reason, the years in the edf and corresponding txt files can be different, so just ignore the date part. But we need to include a date in order to calculate the time difference in seconds, so use today's date.
    #print("Stage: ", a, "| Raw: ", b, " | Stage - raw = ", diff)

    assert(diff >= 0)
    diffPoints = diff * fs
    y = rawData[diffPoints: None]
    return y, diff

def customFilter(y, fs):
    y2 = y
    # Downsample
    print(f'Original fs: {fs}, length: {len(y2)}')
    if fs < 128:
        y2 = mne.filter.resample(y2, up = 128/fs, down = 1, verbose = False)
    elif fs > 128:
        y2 = mne.filter.resample(y2, up = 1, down = fs/128, verbose = False)
    fs = 128
    # 50Hz notch filter
    # Passband filter 0.01-50Hz.
    print(f'Intermediate fs: {fs}, length: {len(y2)}')
    y2 = filter_data(y2, sfreq = fs, l_freq = 0.5, h_freq = 50, l_trans_bandwidth = 0.01, h_trans_bandwidth = 0.01, method = 'fir', fir_window = 'hamming', phase = 'zero', fir_design = 'firwin', verbose = False)
    print(f'New fs: {fs}, length: {len(y2)}')

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
            
        
    #print(f'len1 {len(epochData)}, len2 {len(epochData2)}')
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
    #print(f"Stages: W {num_W} S1 {num_S1} S2 {num_S2} S3 {num_S3} S4 {num_S4} R {num_R}")
    return ({'W': num_W, 'S1': num_S1, 'S2': num_S2, 'S3': num_S3, 'S4': num_S4, 'R': num_R, 'All': num_all})

def calculate_epochs_between_times(startTime, endTime):
    a = datetime.strptime(startTime, '%H:%M:%S')
    b = datetime.strptime(endTime, '%H:%M:%S')
    seconds_passed = (b - a).total_seconds()
    if seconds_passed < 0:
        seconds_passed += 86400 # seconds in a day
    numEpochs = int(seconds_passed) / int(30)
    return int(numEpochs)

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
    # print(f'Raw info: {raw.info}')
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
    startRecordTime = datetime.strptime(stageDate + ' ' + stageTime, '%d/%m/%Y %H:%M:%S')
    #print(f"Start time: {startTime}")

    # Calculate starting point in Raw eeg data
    y, diff = trimStartCAP(rawData, startRecordTime, rawTimestamp, fs)
    y2, fs = customFilter(y, fs)    # returns new frequency of 128
    dataPointsInEpoch = fs * 30
    print(f'data points total: {len(y2)}')
    sleepDict = {'SLEEP-S0': 'W', 'SLEEP-S1': 'S1', 'SLEEP-S2': 'S2', 'SLEEP-S3': 'S3', 'SLEEP-S4': 'S4', 'SLEEP-REM': 'R'}
    i = 0
    epochData = []

    # Append the first stage, assuming it is properly labelled
    stage = re.search(r'SLEEP-(REM|S0|S1|S2|S3|S4)', stageLines[22]).group()
    startTime = re.search('\d{2}:\d{2}:\d{2}', stageLines[22]).group()
    start = i*dataPointsInEpoch
    end = (i + 1) * dataPointsInEpoch
    amplitudeData = y2[start: end]
    epochData.append([sleepDict[stage], amplitudeData])
    prevTime = startTime
    i += 1

    for line in stageLines[23: None]:
        stage = re.search(r'SLEEP-(REM|S0|S1|S2|S3|S4)', line)
        #print(line)
        # Ignore lines with MCAP events
        if stage is None:
            pass
        else:
            curTime = re.search('\d{2}:\d{2}:\d{2}', line).group()
            unlabelled_epochs = calculate_epochs_between_times(prevTime, curTime) - 1
            # Account for unlabelled epochs
            if unlabelled_epochs != 0:
                #print(f'Unlabelled epochs = {unlabelled_epochs} for line {line}')
                for j in range(unlabelled_epochs):
                    start = i*dataPointsInEpoch
                    end = (i + 1) * dataPointsInEpoch
                    amplitudeData = y2[start: end]
                    prev_sleepDict = epochData[-1][0]   # Use last known epoch sleep label for unlabelled epoch
                    #print(f'Appending {prev_sleepDict}')
                    epochData.append([prev_sleepDict, amplitudeData])
                    i += 1
            stage = stage.group()
            
            start = i*dataPointsInEpoch
            end = (i + 1) * dataPointsInEpoch
            amplitudeData = y2[start: end]
            
            # If last annotated epoch was cut off early (epoch duration < 30s), cut off last epoch
            if end > len(y2):
                #print(f'len y2 {len(y2)} end {end}')
                #print("Last epoch cut off! Breaking out of loop")
                break
            i += 1
            
            epochData.append([sleepDict[stage], amplitudeData])
            # Get timestamp of current epoch
            endTime = re.search(r'\d{2}:\d{2}:\d{2}', line).group() 
            prevTime = curTime
    #print(endTime)
    # Add 30s to endtime to denote when epoch recording stopped
    # endTime = datetime.strptime(endTime, '%H:%M:%S')
    # delta = timedelta(seconds = 30)
    # endTime += delta
    # endTime = endTime.strftime('%H:%M:%S')
    numEpochs = calculate_epochs_between_times(startTime, endTime) + 1
    #print(numEpochs, i)
    print(endTime)
    assert(numEpochs == i)


    #print(endTime)
    return epochData, dataPointsInEpoch, pID, pClass, startTime, endTime, numEpochs

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
    startTime = lines[1][-20:-1]
    startTime = datetime.strptime(startTime, '%d.%m.%Y %H:%M:%S')


    
    # Power spectrum
    y, offset = trimStartBerlin(rawData, startTime, rawTimestamp, dataPointsInEpoch, fs, lines)
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
dataset = "CAP"
destName = 'CAP_Overlap'
overlapBool = True

print(f"Dataset: {dataset} Overlap: {overlapBool}")


if dataset == 'Berlin':
    rawDir = 'F:\Thesis B insomnia data\Insomnia data\Data_Study3\Berlin PSG EDF files'
    stageDir = 'F:\\Thesis B insomnia data\\Insomnia data\\Data_Study3\\Berlin PSG annotation files'
    destDir = 'F:\\Berlin data formatted'
    dest_file_h5 = 'F:/Sleep data formatted/Berlin.h5'
    for rawName, stageName, p in zip(os.listdir(rawDir), os.listdir(stageDir), range(67)):
        epochData, dataPointsInEpoch, pID, pClass = annotateDataBerlin(rawDir, stageDir, rawName, stageName)
        if overlap == True:
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
    # s = pd.Series(epochData)
    # s.to_hdf(os.path.join(destDir, 'Berlin.h5'), key = str(pID))




elif dataset == 'CAP':
    rawDir = 'F:/CAP Sleep Database/'
    dest_file_h5 = f'F:/Sleep data formatted/{destName}.h5'
    rawFiles = ['ins1.edf', 'ins2.edf', 'ins3.edf', 'ins4.edf', 'ins5.edf', 'ins6.edf', 'ins7.edf', 'ins8.edf', 'ins9.edf', 'n1.edf', 'n2.edf', 'n3.edf', 'n4.edf', 'n5.edf', 'n10.edf', 'n11.edf', 'n12.edf', 'n14.edf']
    stageFiles = ['ins1.txt', 'ins2.txt', 'ins3.txt', 'ins4.txt', 'ins5.txt', 'ins6.txt', 'ins7.txt', 'ins8.txt', 'ins9.txt', 'n1.txt', 'n2.txt', 'n3.txt', 'n4.txt', 'n5.txt', 'n10.txt', 'n11.txt', 'n12.txt', 'n14.txt']
    #stageFiles = ['n12.txt']
    #rawFiles = ['n12.edf']

    metadata_list_original = []
    metadata_list_overlap = []
    for rawF, stageF in  zip(rawFiles, stageFiles):

        epochData, dataPointsInEpoch, pID, pClass, startTime, endTime, numEpochs = annotateDataCAP(rawF, stageF)
        (sleep_stage_count) = countSleepStages([epoch[0] for epoch in epochData]) # returns W, S1, ... R, total/all
        metadata_original = (pID, pClass, startTime, endTime, (sleep_stage_count))
        if overlapBool == True:
            epochData = overlap(epochData, dataPointsInEpoch)
            (sleep_stage_count) = countSleepStages([epoch[0] for epoch in epochData]) # returns W, S1, ... R, total/all
            metadata_overlap = (pID, pClass, startTime, endTime, (sleep_stage_count))
            metadata_list_overlap.append(metadata_overlap)
        else:
            metadata_overlap = None
        metadata_list_original.append(metadata_original)
        # epochData = removeDCOffset(epochData)
        # epochData = scale(epochData)
        # epochData = removeArtefacts(epochData)  # We need to do this last to ensure when we scale the data later, all values lie within [-1, 1]
        #print(sleep_stage_count)
        dataCols = [f'X{x}' for x in range(1, dataPointsInEpoch + 1)]
        columns = ['Sleep_Stage', *dataCols]
        data = [[str(epoch[0]), *epoch[1]] for epoch in epochData]

        df = pd.DataFrame(columns = columns, data = data)

        

        store = pd.HDFStore(dest_file_h5)
        store.put(pID, df)
        
        store.get_storer(pID).attrs.metadata = {'original': metadata_original, 'overlap': metadata_overlap}
        store.close()


    df_meta_original = pd.DataFrame(columns = ['pID', 'pClass', 'Start time', 'End time', 'Sleep stage count'], data = metadata_list_original)
    df_meta_original.to_csv(f'F:\Sleep data formatted\{destName}_Original.csv')
    if overlapBool == True:
        df_meta_overlap = pd.DataFrame(columns = ['pID', 'pClass', 'Start time', 'End time', 'Sleep stage count'], data = metadata_list_overlap)
        df_meta_overlap.to_csv(f'F:\Sleep data formatted\{destName}_Overlap.csv')

# %%

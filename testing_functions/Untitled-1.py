# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydot_ng as pydot

from datetime import datetime

### CHANGE THESE
dataset = 'CAP_all_control_insomnia'
balance = True
subdataset = 'REM'
method = 'inter' # inter or intra
###

dataPath = f'F:\\Sleep data formatted\\{dataset}.h5'
numEpochDataPoints = 128*30
if dataset == 'CAP_all_control_insomnia':
    pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n1', 'n2', 'n3', 'n4', 'n5', 'n10', 'n11', 'n12', 'n14']
    # Exclude n10, n11, 14
    # pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'n1', 'n2', 'n3', 'n4', 'n5', 'n12']
    # For GroupKFold CV
    group_dict = {
        'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4, 'n5': 5, 'n10': 6, 'n11': 7, 'n12': 8, 'n14': 9,
        'ins1': 1, 'ins2': 2, 'ins3': 3, 'ins4': 4, 'ins5': 5, 'ins6': 6, 'ins7': 7, 'ins8': 8, 'ins9': 9
    }
    # group_dict = {
    #    'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4, 'n5': 5, 'n12': 6,
    #    'ins1': 1, 'ins2': 2, 'ins3': 3, 'ins4': 4, 'ins5': 5, 'ins6': 6}
    n_splits = 9
elif dataset == 'Berlin':
    pIDs = [
        # Insomnia
        1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75,
        # Controls
        3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65
    ]
    group_dict = {}
    n_splits = 5



# We need to split the dataset into sleep stages
X = []
y = []
groups = []
num_insomnia = 0
num_control = 0

max_ins_patients = 30
max_con_patients = 30
ins_patients = 0
con_patients = 0


pClass_dict = {'G': 0, 'I': 1}

pIDs = ['ins1']
for pID in pIDs:
    pID = str(pID)
    print(f'pID: {pID}')
    # To avoid loading all data into RAM, we load only one patient at a time
    with pd.HDFStore(dataPath) as store:
        df = store[pID]
        pID, pClass, startTime, endTime, original_fs, W, S1, S2, S3, S4, R, total = (store.get_storer(pID).attrs.metadata)['overlap'] # Change to overlap or original



# %%
a = (24 + 72 + 34) * 6 + 3
print(a)
# %%

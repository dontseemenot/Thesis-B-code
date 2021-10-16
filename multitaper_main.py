# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mne.time_frequency import tfr_array_multitaper
from datetime import datetime
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("specific_dataset", help = "Specific dataset name. Choose from CAP_overlap, Berlin_no_overlap.")
    parser.add_argument("pID", help = "patient ID")
    return parser

cmdline = True
if cmdline == True:
    parser = get_parser()
    args = vars(parser.parse_args())
else:
    # Run program and specify parameters in notebook
    args = {}
    args['specific_dataset'] = 'CAP_no_overlap_filter2'
    args['pID'] = "ins1"


# if args['dataset'] == 'CAP':
#     pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n1', 'n2', 'n3', 'n4', 'n5', 'n10', 'n11', 'n12', 'n14']

# elif args['dataset'] == "Berlin":
#     pIDs = [
#         'IM_01 I', 'IM_02 I', 'IM_03 G', 'IM_04 I', 'IM_05 I', 'IM_06 I', 'IM_07 G', 'IM_08 G', 'IM_09 G', 'IM_10 G', 'IM_11 G', 'IM_12 G', 'IM_15 I', 'IM_16 I', 'IM_17 I', 'IM_18 I', 'IM_19 I', 'IM_20 I', 'IM_21 I', 'IM_22 G', 'IM_24 G', 'IM_26 I', 'IM_27 I', 'IM_28 G', 'IM_29 G', 'IM_30 G', 'IM_31 G', 'IM_32 G', 'IM_33 G', 'IM_34 G', 'IM_35 G', 'IM_36 G', 'IM_38 G', 'IM_40 G', 'IM_41 I', 'IM_42 I', 'IM_43 I', 'IM_44 G', 'IM_45 G', 'IM_46 G', 'IM_47 G', 'IM_48 G', 'IM_49 G', 'IM_50 G', 'IM_51 G', 'IM_52 I', 'IM_53 I', 'IM_54 I', 'IM_55 I', 'IM_56 I', 'IM_57 I', 'IM_59 G', 'IM_60 I', 'IM_61 G', 'IM_62 I', 'IM_63 I', 'IM_64 I', 'IM_65 G', 'IM_66 I', 'IM_68 I', 'IM_69 I', 'IM_70 I', 'IM_71 I', 'IM_73 I', 'IM_74 I', 'IM_75 I', 'IM_39 G'
#     ]


# Multitaper spectrogram
# data = np.reshape(data1, (1, 1, 3840))
sfreq = 128
f_min = 0.1
f_max = 25.6
freqs = np.linspace(f_min, f_max, num = 256, endpoint = True)

N = 6
n_cycles = freqs    # Number of cycles in wavelet per frequency interval
delta_f = 1
time_bandwidth = N * delta_f   # 6 - 1 = 5 tapers

my_dpi = 331
matplotlib.rcParams['savefig.dpi'] = my_dpi
matplotlib.rcParams["figure.figsize"] = (1, 1.005)  # for terminal

start = datetime.now()
dt_string_start = start.strftime("%d-%m-%Y %H.%M.%S")
x = np.arange(1, 3841, 1)

destination = f"./data/{args['specific_dataset']}/tfr/"
print(f"Processing {args['pID']}...")

# Load data
with pd.HDFStore(f"./data/{args['specific_dataset']}/{args['specific_dataset']}.h5", mode = 'r') as store:
    data = store[args['pID']]
    data = np.array(data)
    indices = np.arange(0, data.shape[0], 1)
    #indices = np.arange(0, data.shape[0], 100)  # for debugging
    #indices = np.arange(579, 580, 100) # for debugging
    for i in indices:
        epoch = data[i][1:]
        stage = data[i][0]
        epoch = epoch.reshape((1, 1, epoch.shape[0]))
        power = tfr_array_multitaper(epoch, sfreq = sfreq, freqs = freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, output = 'power')
        plt.pcolormesh(x, freqs, power[0][0], shading = 'flat', cmap = 'gray')
        plt.axis('off')
        plt.savefig(f"{destination}{args['pID']}-{i}-{stage}.png", bbox_inches='tight', transparent=True, pad_inches=0)
        plt.clf()
        plt.close()
        if i % 100 == 0:
            print(f"epoch {i} done")


end = datetime.now()
dt_string_end = end.strftime("%d-%m-%Y %H.%M.%S")
print(f"Start: {dt_string_start} End: {dt_string_end} Took: {(end - start).total_seconds()} seconds")

# %%

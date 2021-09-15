'''
dataset: name of the dataset to use
balance: if dataset balancing is applied (max 800 epochs per patient)
subdataset: choose which subdataset to utilise
results_dir: directory to store results
method: type of training and testing paradigm
'''


from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H.%M.%S")

# Parameters for model training and testing
title = 'testing excel saving'
dataset = 'CAP'
specific_dataset = 'CAP_overlap'
balance = True
subdataset = 'SWS'  # ALL, LSS, SWS, REM
results_dir = f'./{dataset} results/{title} {dt_string}/'
method = 'intra' # Choose from inter, intra
additional_info = 'Testing'
model_name = 'AlexNet'
outer_fold_limit = 2    # For debugging purposes; set to > n_outer_split for normal usage

# Dataset
max_ins_patients = 30
max_con_patients = 30
# Constant model parameters
batch_size = 256
# num_iterations = 80 # epoch count (named as iteration to avoid with epoch samples)
# min_delta = 0.001
# patience = 3
n_inner_split = 3 # Change accordingly if inter or intra patient cv
n_outer_split = 9

tuning = True
# Hyperparameters
param_grid = {
    'lr': [0.0001],
    'C': [0.001],
    'epochs': [3, 6]
}


dataPath = f'./data/{specific_dataset}.h5'    # File of dataset

if dataset == 'CAP':
    pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n1', 'n2', 'n3', 'n4', 'n5', 'n10', 'n11', 'n12', 'n14']
    # Exclude n10, n11, 14
    # pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'n1', 'n2', 'n3', 'n4', 'n5', 'n12']
    # For GroupKFold CV
    group_dict = {
        'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4, 'n5': 5, 'n10': 6, 'n11': 7, 'n12': 8, 'n14': 9,
        'ins1': 1, 'ins2': 2, 'ins3': 3, 'ins4': 4, 'ins5': 5, 'ins6': 6, 'ins7': 7, 'ins8': 8, 'ins9': 9
    }
    n_splits = 9
elif dataset == 'Berlin':
    pIDs = [
        'IM_01 I', 'IM_02 I', 'IM_03 G', 'IM_04 I', 'IM_05 I', 'IM_06 I', 'IM_07 G', 'IM_08 G', 'IM_09 G', 'IM_10 G', 'IM_11 G', 'IM_12 G', 'IM_15 I', 'IM_16 I', 'IM_17 I', 'IM_18 I', 'IM_19 I', 'IM_20 I', 'IM_21 I', 'IM_22 G', 'IM_24 G', 'IM_26 I', 'IM_27 I', 'IM_28 G', 'IM_29 G', 'IM_30 G', 'IM_31 G', 'IM_32 G', 'IM_33 G', 'IM_34 G', 'IM_35 G', 'IM_36 G', 'IM_38 G', 'IM_39 G', 'IM_40 G', 'IM_41 I', 'IM_42 I', 'IM_43 I', 'IM_44 G', 'IM_45 G', 'IM_46 G', 'IM_47 G', 'IM_48 G', 'IM_49 G', 'IM_50 G', 'IM_51 G', 'IM_52 I', 'IM_53 I', 'IM_54 I', 'IM_55 I', 'IM_56 I', 'IM_57 I', 'IM_59 G', 'IM_60 I', 'IM_61 G', 'IM_62 I', 'IM_63 I', 'IM_64 I', 'IM_65 G', 'IM_66 I', 'IM_68 I', 'IM_69 I', 'IM_70 I', 'IM_71 I', 'IM_73 I', 'IM_74 I'    # IM_75 I removed
    ]
    n_splits = 10
    group_dict = {}
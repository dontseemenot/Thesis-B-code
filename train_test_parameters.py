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
dataset = 'CAP_all_control_insomnia'
balance = True
subdataset = 'ALL'  # ALL, LSS, SWS, REM
results_dir = f'./CAP results/compare validation and train acc {dt_string}/'
method = 'intra' # Choose from inter, intra
additional_info = 'Dropout after all CNN, 2 dense layers, input'

# Dataset
max_ins_patients = 30
max_con_patients = 30
# Constant model parameters
batch_size = 256
num_iterations = 100 # epoch count (named as iteration to avoid with epoch samples)
min_delta = 0.001
patience = 3
n_inner_split = 5 # Change accordingly if inter or intra patient cv
n_outer_split = 10


# Hyperparameters
param_grid = {
    'lr': [0.0001],
    'C': [0.001]
}


dataPath = f'F:\\Sleep data formatted\\{dataset}.h5'    # File of dataset

if dataset == 'CAP_all_control_insomnia':
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
        # Insomnia
        1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75,
        # Controls
        3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65
    ]
    group_dict = {}
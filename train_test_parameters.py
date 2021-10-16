'''
dataset: name of the dataset to use
balance: if dataset balancing is applied (max 800 epochs per patient)
subdataset: choose which subdataset to utilise
results_dir: directory to store results
method: type of training and testing paradigm
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help = "Give a name for the job and output file")
    parser.add_argument("dataset", help = "Overall dataset name. Choose from CAP, Berlin.")
    parser.add_argument("specific_dataset", help = "Specific dataset name. Choose from CAP_overlap, Berlin_no_overlap.")
    parser.add_argument("subdataset", help = "Choose epoch subdataset. Choose from ALL, LSS, SWS, REM.")
    parser.add_argument("balance", help = "To apply dataset balancing or not. Choose from True, False.")
    parser.add_argument("method", help = "Which patient testing paradigm to use. Choose from inter, intra.")
    parser.add_argument("model_name", help = "Which model to use. Choose from AlexNet_1D")
    parser.add_argument("batch_size", type = int, help = "Batch size. Choose 256 as default.")
    parser.add_argument("n_splits", type = int, help = "Number of splits for corss validation. For inter-patient method choose n_splits such that n_splits % num_patients == 0")
    parser.add_argument("param_grid", help = "Hyperparameters to use. {\"C\": [0.0001, 0.001, 0.01], \"lr\": [0.0001, 0.001, 0.01], \"epochs\": [60, 80, 100]}")
    return parser
import sys
import json
import argparse



cmdline = True
if cmdline == True:
    parser = get_parser()
    args = vars(parser.parse_args())
    args['balance'] = ("True" == args['balance'])
    args['param_grid'] = json.loads(args['param_grid'])

else:
    # Run program and specify parameters in notebook
    args = {}
    args['title'] = 'Test job debug'
    args['dataset'] = 'CAP'
    args['specific_dataset'] = 'CAP_overlap_filter2'
    args['subdataset'] = 'SWS'
    args['balance'] = True
    args['method'] = 'inter'
    args['model_name'] = 'AlexNet_1D'
    args['batch_size'] = 256
    args['n_splits'] = 9
    args['param_grid'] = {
        'lr': 0.0001,
        'C': 0.0001,
        'epochs': 80
    }



batch_size = 256
numEpochDataPoints = 128*30 
tuning = True
path_data_folder = f'./data/{args["specific_dataset"]}/'    # File of dataset
file_metadata = f"{path_data_folder}{args['specific_dataset']}_metadata.csv"
file_h5 = f"{path_data_folder}{args['specific_dataset']}.h5"
path_tfr_folder = f"{path_data_folder}tfr/"

if args['dataset'] == 'CAP':
    pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n1', 'n2', 'n3', 'n4', 'n5', 'n10', 'n11', 'n12', 'n14']
    # Exclude n10, n11, 14
    # pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'n1', 'n2', 'n3', 'n4', 'n5', 'n12']
    # For GroupKFold CV
    group_dict = {
        'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4, 'n5': 5, 'n10': 6, 'n11': 7, 'n12': 8, 'n14': 9,
        'ins1': 1, 'ins2': 2, 'ins3': 3, 'ins4': 4, 'ins5': 5, 'ins6': 6, 'ins7': 7, 'ins8': 8, 'ins9': 9
    }
elif args['dataset'] == 'Berlin':
    pIDs = [
        'IM_01 I', 'IM_02 I', 'IM_03 G', 'IM_04 I', 'IM_05 I', 'IM_06 I', 'IM_07 G', 'IM_08 G', 'IM_09 G', 'IM_10 G', 'IM_11 G', 'IM_12 G', 'IM_15 I', 'IM_16 I', 'IM_17 I', 'IM_18 I', 'IM_19 I', 'IM_20 I', 'IM_21 I', 'IM_22 G', 'IM_24 G', 'IM_26 I', 'IM_27 I', 'IM_28 G', 'IM_29 G', 'IM_30 G', 'IM_31 G', 'IM_32 G', 'IM_33 G', 'IM_34 G', 'IM_35 G', 'IM_36 G', 'IM_38 G', 'IM_40 G', 'IM_41 I', 'IM_42 I', 'IM_43 I', 'IM_44 G', 'IM_45 G', 'IM_46 G', 'IM_47 G', 'IM_48 G', 'IM_49 G', 'IM_50 G', 'IM_51 G', 'IM_52 I', 'IM_53 I', 'IM_54 I', 'IM_55 I', 'IM_56 I', 'IM_57 I', 'IM_59 G', 'IM_60 I', 'IM_61 G', 'IM_62 I', 'IM_63 I', 'IM_64 I', 'IM_65 G', 'IM_66 I', 'IM_68 I', 'IM_69 I', 'IM_70 I', 'IM_71 I', 'IM_73 I', 'IM_74 I'    # IM_75 I, 39 G removed due to lack of R stages
    ]
    group_dict = {pID: i % args['n_splits'] for i, pID in enumerate(pIDs)}


# For debugging purposes
max_ins_patients = 100 
max_con_patients = 100
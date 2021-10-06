# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydot_ng as pydot
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import json
from tensorflow.keras.callbacks import EarlyStopping

from train_test_helpers import *
from train_test_model import *
from train_test_parameters import *
import sys
import PIL
import re

np.random.seed(42)


# We need to split the dataset into sleep stages
X = []
y = []
groups = []
num_insomnia = 0
num_control = 0


ins_patients = 0
con_patients = 0
# Berlin. can shuffle if desired
#insomniaIDs = [1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75]
#goodIDs = [3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65]

pClass_dict = {'G': 0, 'I': 1}
print("Loading patient data...")
for pID in pIDs:
    pID = str(pID)
    with pd.HDFStore(f'{dataPath}/{args["specific_dataset"]}.h5', mode = 'r') as store: # set to read only
        df = store[pID]
        # Exclude 'Other' stages
        df = df.loc[(df['Sleep_Stage'] == 'W') | (df['Sleep_Stage'] == 'S1') | (df['Sleep_Stage'] == 'S2') | (df['Sleep_Stage'] == 'S3') | (df['Sleep_Stage'] == 'S4') | (df['Sleep_Stage'] == 'R') ]
        pID, pClass, startTime, endTime, original_fs, fs, W, S1, S2, S3, S4, R, other, total = (store.get_storer(pID).attrs.metadata)['metadata'] # Change to overlap or original
        stages = {'W': W, 'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'R': R, 'Other': other}
        print(f'{pID} ', stages)
        if pClass == 'I' and ins_patients == max_ins_patients or pClass == 'G' and con_patients == max_con_patients:
            print(f'Rejected pid {pID}, limit reached')
        else:
            threshold = max_num_epochs(stages)
            # Choose from All, LSS, SWS, REM, BSL
            if args['balance'] == True:
                df = balance_dataset(df, threshold, args['subdataset'])   # Balance distribution of 5 types of subdataset
            else:
                df = get_sleep_epochs(df, args['subdataset'])
            if pClass == 'I':
                num_insomnia += len(df)
                ins_patients += 1
            else:
                num_control += len(df)
                con_patients += 1
            # Because threshold can be different size, we need to append epoch data one by one to list
            if args['model_name'] == "AlexNet_1D":
                [X.append(row) for row in df.iloc[:, 1:None].to_numpy()]
                [y.append(pClass_dict[pClass]) for i in range(len(df.iloc[:, 0]))]
                [groups.append(group_dict[pID]) for i in range(len(df.iloc[:, 0]))]
            elif args['model_name'] == "AlexNet_2D":
                tfr_path = f"{dataPath}tfr/"
                epoch_nums = list(df.index.values) # If balance = true, returns epoch number so we know which image to look for
                for num in epoch_nums:
                    [filename for filename in os.listdir('.') if filename.startswith(f"{pID}-{num}")]


# %%
tfr_path = f"{dataPath}tfr/"
for image in os.listdir(tfr_path):
    epoch2D = tf.keras.preprocessing.image.load_img(f"{tfr_path}{image}")
    epoch2D = tf.keras.preprocessing.image.img_to_array(epoch2D)
    
                    

# %%
print("All patient data loaded")
X, y, groups = custom_preprocess(X, y, groups)
print(f"Subdataset {args['subdataset']}: X.shape {X.shape} Y.shape {y.shape} group.shape {groups.shape} I {num_insomnia} G {num_control}")
# %%

# Make directories for fold information, models, performance metrics, images, test and predicted data
now = datetime.now()
dt_string_start = now.strftime("%d-%m-%Y %H.%M.%S")
results_dir, models_dir, images_dir = create_dirs(args['title'], dt_string_start)

spreadsheet_file = f'{results_dir}Result summary.xlsx'

strategy = tf.distribute.MirroredStrategy()
total_batch_size = batch_size * strategy.num_replicas_in_sync
print(f'Number of devices: {strategy.num_replicas_in_sync}\nBatch size per GPU: {batch_size}\nTotal batch size: {total_batch_size}')

performance_metrics_all = []
results_all = []
cm_all = []

if args['method'] == 'inter':
    cv_outer = GroupKFold(n_splits = args['n_splits'])
elif args['method'] == 'intra':
    cv_outer = KFold(n_splits = args['n_splits'], shuffle = True)

if args['model_name'] == 'AlexNet_1D':
    build_fn = create_model_AlexNet


X_train, y_train, X_test, y_test, groups_train, groups_test, data_info = get_train_test(X, y, groups, cv_outer)
print(data_info)
save_test_info(args, data_info, spreadsheet_file)
# %%
# HYPERPARAMETER TUNING VIA GRIDSEARCHCV
if args['method'] == 'inter':
    cv_inner = GroupKFold(n_splits = args['n_splits'] - 1)
elif args['method'] == 'intra':
    cv_inner = KFold(n_splits = args['n_splits'] - 1, shuffle = True)

model = KerasClassifier(build_fn = build_fn, batch_size = total_batch_size)
search = GridSearchCV(estimator = model, param_grid = args['param_grid'], n_jobs = 1, refit = True, cv = cv_inner, scoring = 'accuracy') # cross-entropy loss
search.fit(X_train, y_train, groups_train)    # Groups necessary only if inter-patinet used
cv_results = search.cv_results_

best_model = search.best_estimator_
best_hyperparams = search.best_params_
save_validation_results(cv_results, args['n_splits'], spreadsheet_file)
# %%

# Retrain model using best hyperparameters found on trainval dataset
best_model.fit(X_train, y_train, validation_data = (X_test, y_test))
best_model.model.save(f'{models_dir}/Best model.h5')
train_acc = best_model.model.history.history['sparse_categorical_accuracy']
train_loss = best_model.model.history.history['loss']
val_acc = best_model.model.history.history['val_sparse_categorical_accuracy']
val_loss = best_model.model.history.history['val_loss']

plot_train_val_acc_loss(val_acc, val_loss, train_acc, train_loss, images_dir)


y_pred = best_model.predict(X_test)
results = {}
results['y_pred'] = y_pred # Convert softmax output to 0 or 1
results['y_test'] = y_test
# np.save(f'{test_pred_dir}/Fold {i} y_pred y_test.npy', results)
# results_all.append(results)

# Confusion matrix
cm = plot_cm(y_pred, y_test, images_dir)
# cm_all.append(cm)

performance_metrics = calculate_performance_metrics(y_test, y_pred, cm)

# save_fold_info(i, hyp_results, performance_metrics, spreadsheet_file)
save_test_results(performance_metrics, best_hyperparams, spreadsheet_file)
# performance_metrics_all.append(performance_metrics)

# plot_fold_test(y_pred, y_test, i, images_dir)

now = datetime.now()
dt_string_end = now.strftime("%d-%m-%Y %H.%M.%S")
print(f"All testing completed with test accuracy: {performance_metrics['accuracy']}\nStart: {dt_string_start} End: {dt_string_end}")
# # %%
# # Average performance over all folds
# performance_metrics_mean = {
#     'Accuracy': np.mean([x['accuracy'] for x in performance_metrics_all]),
#     'Precision': np.mean([x['precision'] for x in performance_metrics_all]),
#     'Recall': np.mean([x['recall'] for x in performance_metrics_all]),
#     'Sensitivity': np.mean([x['sensitivity'] for x in performance_metrics_all]),
#     'Specificity': np.mean([x['specificity'] for x in performance_metrics_all]),
#     'F1': np.mean([x['f1'] for x in performance_metrics_all])
# }
# save_mean_results(performance_metrics_mean, spreadsheet_file)

# %%

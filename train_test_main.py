# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydot_ng as pydot
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import json
from keras.callbacks import EarlyStopping

from algo_helpers import *
from algo_model import *
from parameters import *

np.random.seed(42)
numEpochDataPoints = 128*30


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

for pID in pIDs:
    pID = str(pID)
    with pd.HDFStore(dataPath) as store:
        df = store[pID]
        pID, pClass, startTime, endTime, original_fs, W, S1, S2, S3, S4, R, total = (store.get_storer(pID).attrs.metadata)['overlap'] # Change to overlap or original
        stages = {'W': W, 'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'R': R}
        if pClass == 'I' and ins_patients == max_ins_patients or pClass == 'G' and con_patients == max_con_patients:
            print(f'Rejected pid {pID}, limit reached')
        else:
            threshold = max_num_epochs(stages)
              # Choose from All, LSS, SWS, REM, BSL
            if balance == True:
                df = balance_dataset(df, threshold, subdataset)   # Balance distribution of 5 types of subdataset
            if pClass == 'I':
                num_insomnia += len(df)
                ins_patients += 1
            else:
                num_control += len(df)
                con_patients += 1
            # Because threshold can be different size, we need to append epoch data one by one to list
            [X.append(row) for row in df.iloc[:, 1:None].to_numpy()]
            [y.append(pClass_dict[pClass]) for i in range(len(df.iloc[:, 0]))]
            [groups.append(group_dict[pID]) for i in range(len(df.iloc[:, 0]))]

X, y, groups = custom_preprocess(X, y, groups)
print(f'Subdataset {subdataset}: X.shape {X.shape} Y.shape {y.shape} group.shape {groups.shape} I {num_insomnia} G {num_control} ')


# Make directories for fold information, models, performance metrics, images, test and predicted data
(models_dir, performance_dir, images_dir, test_pred_dir, fold_info_dir) = create_dirs(results_dir)

spreadsheet_file = f'{results_dir}Testing.xlsx'
test_info = {
    'Start time': dt_string,
    'Dataset': dataset,
    'Subdataset': subdataset,
    'Test method': method,
    'Balancing': balance,
    'Batch size': batch_size,
    'Iterations': num_iterations,
    'min_delta': min_delta,
    'patience': patience,
    'n_inner_split': n_inner_split,
    'n_outer_split': n_outer_split,
    'Hyperparameters': param_grid,
    'Additional info': additional_info
}
save_test_info(test_info, spreadsheet_file)

# %%
performance_metrics_all = []
results_all = []
cm_all = []
# K-fold into train and test datasets
if method == 'inter':
    cv_outer = GroupKFold(n_splits = n_outer_split)
elif method == 'intra':
    cv_outer = KFold(n_splits = n_outer_split, shuffle = True)
# Outer Nested CV

for (train_index, test_index), i in zip(cv_outer.split(X, y, groups), range(5)):
    print(f'Fitting Fold {i}...')
    X_train, y_train, X_test, y_test, groups_train, groups_test, info = get_train_test(X, y, groups, train_index, test_index)
    print(info)

    if method == 'inter':
        cv_inner = GroupKFold(n_splits = n_inner_split)
    elif method == 'intra':
        cv_inner = KFold(n_splits = n_inner_split, shuffle = True)

    # HYPERPARAMETER TUNING VIA GRIDSEARCHCV
    AlexNet = KerasClassifier(build_fn = create_model_AlexNet, epochs = num_iterations, batch_size = batch_size)
    search = GridSearchCV(estimator = AlexNet, param_grid = param_grid, n_jobs = 1, refit = True, cv = cv_inner, scoring = 'neg_log_loss') # cross-entropy loss
    search_result = search.fit(X_train, y_train, groups = groups_train)
    val_acc = search_result.best_estimator_.model.history.history['sparse_categorical_accuracy']
    val_loss = search_result.best_estimator_.model.history.history['loss']

    best_hyp = search_result.best_params_
    C = search_result.best_params_['C']
    lr = search_result.best_params_['lr']
    hyp_results = {
        'parameters': search.cv_results_['params'],
        'loss': abs(search.cv_results_['mean_test_score']),    # Lower loss is better
        'loss_std': search.cv_results_['std_test_score']
    }

    # NO HYPERPARAMETER TUNING
    # AlexNet = KerasClassifier(build_fn = lambda: create_model_AlexNet(),  epochs = num_iterations, batch_size = batch_size)
    # # early_stopping = EarlyStopping(monitor='neg_log_loss', min_delta = min_delta, patience = patience)
    # # train_result = AlexNet.fit(X_train, y_train, callbacks = [early_stopping])
    # train_result = AlexNet.fit(X_train, y_train, validation_data = (X_test, y_test))
    # train_acc = train_result.model.history.history['sparse_categorical_accuracy']
    # train_loss = train_result.model.history.history['loss']
    # val_acc = train_result.model.history.history['val_sparse_categorical_accuracy']
    # val_loss = train_result.model.history.history['val_loss']
    # plot_train_val_acc_loss(i, val_acc, val_loss, train_acc, train_loss, images_dir)
    # Test data

    AlexNet.model.save(f'{models_dir}/Fold {i} AlexNet.h5')
    ## Load test data
    # a = np.load(f'{folder}y_pred y_test.npy', allow_pickle = True)
    # y_pred = a.item()['y_pred']
    # y_test = a.item()['y_test']
    ###
    y_pred = AlexNet.predict(X_test)
    results = {}
    results['y_pred'] = y_pred # Convert softmax output to 0 or 1
    results['y_test'] = y_test
    np.save(f'{test_pred_dir}/Fold {i} y_pred y_test.npy', results)
    results_all.append(results)

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Control (0)', 'Insomnia (1)']).plot(cmap = 'Blues')
    plt.title(f'Fold {i} Confusion Matrix')
    plt.savefig(f'{images_dir}/Fold {i} CM.png', dpi = 100)
    plt.clf()
    cm_all.append(cm)

    cm_norm = confusion_matrix(y_test, y_pred, normalize = 'true')
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_norm, display_labels = ['Control (0)', 'Insomnia (1)']).plot(cmap = 'Blues')
    plt.title(f'Fold {i} Confusion Matrix Normalized')
    plt.savefig(f'{images_dir}/Fold {i} Confusion Matrix Normalized.png', dpi = 100)
    plt.clf()

    performance_metrics = {}
    performance_metrics['accuracy'] = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    performance_metrics['precision'] = tp/(tp + fp)
    performance_metrics['recall'] = tp/(tp + fn)
    performance_metrics['sensitivity'] = tp/(tp + fn)
    performance_metrics['specificity'] = tn/(tn + fp)
    performance_metrics['f1'] = 2 * performance_metrics['precision'] * performance_metrics['recall'] / (performance_metrics['precision'] + performance_metrics['recall'])

    # save_fold_info(i, hyp_results, performance_metrics, spreadsheet_file)
    save_fold_info(i, performance_metrics, spreadsheet_file)
    plt.clf()
    performance_metrics_all.append(performance_metrics)

        

    plt.plot(y_pred, label = 'pred', linestyle = 'None', markersize = 1.0, marker = '.')
    plt.plot(y_test, label = 'test')
    plt.title(f'Fold {i} Test vs predicted')
    plt.ylabel('Control (0), Insomnia (1)')
    plt.xlabel('Test epoch')
    plt.legend()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.savefig(f'{images_dir}/Fold {i} Test vs predicted.png', dpi = 200)
    plt.clf()

    print(f"Fold {i} completed with accuracy: {performance_metrics['accuracy']}\n")

# Average performance over all folds
performance_metrics_avg = [
    np.mean([x['accuracy'] for x in performance_metrics_all]),
    np.mean([x['precision'] for x in performance_metrics_all]),
    np.mean([x['recall'] for x in performance_metrics_all]),
    np.mean([x['sensitivity'] for x in performance_metrics_all]),
    np.mean([x['specificity'] for x in performance_metrics_all]),
    np.mean([x['f1'] for x in performance_metrics_all])
]
save_mean_results(performance_metrics_avg, spreadsheet_file)


print(f'All training and testing completed. Average accuracy: {performance_metrics_avg[0]}')
# %%

# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, GroupShuffleSplit, PredefinedSplit, StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from train_test_helpers import *
from train_test_model import *
from train_test_parameters import *
import sys
import PIL
import re
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

np.random.seed(42)


# We need to split the dataset into sleep stages
X = []
y = []
groups = []
num_frames_ins = 0
num_frames_con = 0
num_patients_ins = 0
num_patients_con = 0
# Berlin. can shuffle if desired
#insomniaIDs = [1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75]
#goodIDs = [3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65]
#pIDs = ["ins2"]
pClass_dict = {'G': 0, 'I': 1}
allowed_stages = {
    "ALL": ["W", "S1", "S2", "S3", "S4", "R"],
    "LSS": ["S1", "S2"],
    "N1": ["S1"],
    "N2": ["S2"],
    "SWS": ["S3", "S4"],
    "R": ["R"]
}
print("Loading patient data...")
# %%
# Determine which patients go in which groups
df_metadata = pd.read_csv(file_metadata)
group_dict = group_greedy(df_metadata, pIDs, args['n_splits'])

if args["model_name"] == "AlexNet_1D":
    for pID in pIDs:
        pID = str(pID)
        pClass = df_metadata.loc[df_metadata["pID"] == pID, 'pClass'].values[0]
        # Exclude 'Other' stages
        # df = df.loc[(df['Sleep_Stage'] == 'W') | (df['Sleep_Stage'] == 'S1') | (df['Sleep_Stage'] == 'S2') | (df['Sleep_Stage'] == 'S3') | (df['Sleep_Stage'] == 'S4') | (df['Sleep_Stage'] == 'R') ]
        stage_names = ["W", "S1", "S2", "S3", "S4", "R", "Other", "Total"]
        stage_count = df_metadata.loc[df_metadata["pID"] == pID, 'W':'Total'].values[0]
        stages = {key: value for key, value in zip(stage_names, stage_count)}
        print(f'Loading {pID} ', stages)
        with pd.HDFStore(file_h5, mode = 'r') as store: # set to read only
            df = store[pID]
        if args['balance'] == True:
            threshold = max_num_epochs(stages)
            df = balance_dataset(df, threshold, args['subdataset'])   # Balance distribution of 5 types of subdataset
        df = get_sleep_epochs(df, args['subdataset'])
        if pClass == 'I':
            num_frames_ins += len(df)
            num_patients_ins += 1
        else:
            num_frames_con += len(df)
            num_patients_con += 1
        # Because threshold can be different size, we need to append epoch data one by one to list
        [X.append(row) for row in df.iloc[:, 1:None].to_numpy()]
        [y.append(pClass_dict[pClass]) for i in range(len(df.iloc[:, 0]))]
        [groups.append(group_dict[pID]) for i in range(len(df.iloc[:, 0]))]
elif args['model_name'] == "AlexNet_2D":
    seen_pIDs_ins = []
    seen_pIDs_con = []
    ORIGINAL_SIZE = (256, 256)
    #PRECROPPED_SIZE = (256, 224)
    # PRECROPPED_SIZE = (224, 224)
    CROPPED_SIZE = (224, 224)
    count = 0
    for file_image in os.listdir(path_tfr_folder):
        pID, frame, stage = file_image.split("-")
        stage = stage.split(".")[0] # remove png extension
        if pID in pIDs and stage in allowed_stages[args["subdataset"]]: # Ignore 'Other' and excluded stages from current test
            # print(file_image + " OK")
            img = Image.open(path_tfr_folder + file_image).convert('L')
            img = img.resize(CROPPED_SIZE)
            img = np.array(img) # image to tensor
            
            # img = img.reshape((img.shape[0], img.shape[1], 1))  # tensor
            pClass = pClass_dict[df_metadata.loc[df_metadata["pID"] == pID, 'pClass'].values[0]]
            pClass = pID[-1]
            if pClass == 'I':
                num_frames_ins += 1
                if pID not in seen_pIDs_ins:
                    seen_pIDs_ins.append(pID)
                    num_patients_ins += 1
            else:
                num_frames_con += 1
                if pID not in seen_pIDs_con:
                    seen_pIDs_con.append(pID)
                    num_patients_con += 1

            X.append(img)
            y.append(pClass)
            groups.append(group_dict[pID])
            count += 1
            if count % 10000 == 0:
                print(f"Loaded {count} images")



assert(len(X) == len(y))
assert(len(y) == len(groups))     
print("All patient data loaded")

# X, y, groups = custom_preprocess(X, y, groups)
X = np.asarray(X)   # Convert to numpy array
y = np.asarray(y)       
groups = np.asarray(groups)
print(f"Subdataset {args['subdataset']}: X.shape {X.shape} Y.shape {y.shape} group.shape {groups.shape} I frames:{num_frames_ins} G frames:{num_frames_con} I patients:{num_patients_ins} G patients:{num_patients_con}")


# Make directories for fold information, models, performance metrics, images, test and predicted data
now = datetime.now()
dt_start = now.strftime("%d-%m-%Y %H.%M.%S")
results_dir, models_dir, images_dir = create_dirs(args['title'], dt_start)

spreadsheet_file = f'{results_dir}Result summary.xlsx'

strategy = tf.distribute.MirroredStrategy()
total_batch_size = batch_size * strategy.num_replicas_in_sync
print(f'Number of devices: {strategy.num_replicas_in_sync}\nBatch size per GPU: {batch_size}\nTotal batch size: {total_batch_size}')

performance_metrics_all = []
results_all = []
cm_all = []

if args['model_name'] == 'AlexNet_1D':
    build_fn = create_model_AlexNet_1D
elif args['model_name'] == 'AlexNet_2D':
    build_fn = create_model_AlexNet_2D
    
elif args['model_name'] == 'LeNet_5_1D':
    build_fn = create_model_LeNet_5_1D
elif args['model_name'] == 'LeNet_549_1D':
    build_fn = create_model_LeNet_549_1D


# HYPERPARAMETER TUNING VIA GRIDSEARCHCV
if args['method'] == 'inter':
    cv_outer = GroupKFold(n_splits = args['n_splits'])
    # cv_outer = PredefinedSplit(groups)
    #cv_outer = StratifiedGroupKFold(n_splits = args['n_splits'])
elif args['method'] == 'intra':
    cv_outer = KFold(n_splits = args['n_splits'], shuffle = True)

data_info = {
    'X.shape': str(X.shape),
    'y.shape': str(y.shape),
    'Insomnia': num_frames_ins,
    'Control': num_frames_con
}
save_parameters(args, data_info, spreadsheet_file)

# X, y, groups = custom_preprocess(X, y, groups)
# %%
# Split data into train and test sets

groups_used_for_valid = list(range(0, args["n_splits"]))  # Ensure unique validation set across all train folds

fold_num = 0
for i, (train_valid_index, test_index) in zip(range(100), cv_outer.split(X, y, groups)):

    X_train_valid, y_train_valid, groups_train_valid, X_test, y_test, groups_test = get_train_test2(X, y, groups, train_valid_index, test_index)
    # print(info)
    # Further split train data into train and valid set
    if args['method'] == 'inter':
        cv_inner = GroupKFold(n_splits = args['n_splits'] - 1)
    elif args['method'] == 'intra':
        cv_inner = KFold(n_splits = args['n_splits'] - 1, shuffle = True)
    
    # for train_index, valid_index in cv_inner.split(X_train_valid, y_train_valid, groups_train_valid):
    #     chosen_group = groups_train_valid[valid_index][0]
    #     if chosen_group in groups_used_for_valid:   # To get a unique validation set per train-test split
    #         groups_used_for_valid.remove(chosen_group)
    #         print(f"validation with group = {chosen_group}")
    #         print(f"remaining groups {groups_used_for_valid}")
    #         break

    unique_test_groups = np.unique(groups_test)
    valid_group_num = (unique_test_groups[0] + 1) % args['n_splits']
    # Get indices that belong to train or valid
    print(valid_group_num)

    # Get train indices for training group
    # train_index = np.array([x for i, x in enumerate(train_valid_index) if groups[i] != valid_group_num])
    train_index = []
    for i, group in enumerate(groups_train_valid):
        if group != valid_group_num:
            train_index.append(i)
    train_index = np.array(train_index)
    # Get valid indices for valid group
    valid_index = []
    for i, group in enumerate(groups_train_valid):
        if group == valid_group_num:
            valid_index.append(i)
    valid_index = np.array(valid_index)
    print(train_index.shape, valid_index.shape)
    X_train, y_train, groups_train, X_valid, y_valid, groups_valid = get_train_test2(X_train_valid, y_train_valid, groups_train_valid, train_index, valid_index)

    unique_train_groups = np.unique(groups_train)   # Groups only needed for GroupKFold
    unique_valid_groups = np.unique(groups_valid)
    # unique_test_groups = np.unique(groups_test)

    unique, counts = np.unique(y_train, return_counts = True)
    train_info = dict(zip(unique, counts))
    unique, counts = np.unique(y_valid, return_counts = True)
    valid_info = dict(zip(unique, counts))
    unique, counts = np.unique(y_test, return_counts = True)
    test_info = dict(zip(unique, counts))

    print(unique_train_groups, unique_valid_groups, unique_test_groups)

    epoch_counts = {
        'fold': fold_num,
        'Train groups': str(unique_train_groups),   # To fit list into df
        'Valid groups': str(unique_valid_groups),
        'Test groups': str(unique_test_groups),
        'X_train.shape': str(X_train.shape),
        'y_train.shape': str(y_train.shape),
        'X_valid.shape': str(X_valid.shape),
        'y_valid.shape': str(y_valid.shape),
        'X_test.shape': str(X_test.shape),
        'y_test.shape': str(y_test.shape),
        'Train class count': str(train_info),
        'Valid class count': str(valid_info),
        'Test class count': str(test_info)
    }
    #|print(epoch_counts)
    #

    # Data augmentation
    # if args['GaussianNoise'] == True:
    #     X_train, y_train = augment(X_train, y_train)
    # es = EarlyStopping(monitor = 'loss', mode = 'min', verbose=1, min_delta = )
    # best_model_info = "Epoch:{epoch:02d}\n{sparse_categorical_accuracy:.2f}-{val_sparse_categorical_accuracy:.2f}-{loss:.4f}-{val_loss:.4f}"

    # Save the best model only
    model_path=f"{results_dir}best_model.h5"
    #mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)

    # es = EarlyStopping(monitor = 'val_loss', mode = 'min', min_delta)
    model = KerasClassifier(build_fn = build_fn, C = args['C'], lr = args['lr'], std = args['std'], dropout_cnn = args['dropout_cnn'], batch_size = total_batch_size, epochs = args['epochs'])

    pipe = Pipeline([
        ('standardscaler', StandardScaler()),   # Standardization to zero mean, unit variance
        ('minmaxscaler', MinMaxScaler(feature_range = (-1, 1))), # Normalization
        ('reshapetotensor', ReshapeToTensor()),
    ])
    # We need to pipeline datasets individually, because validation data does not get pipelined
    if args['model_name'] == "AlexNet_1D":
        X_train = pipe.fit_transform(X_train)
        X_valid = pipe.fit_transform(X_valid)
        X_test = pipe.fit_transform(X_test)

    elif args['model_name'] == "AlexNet_2D":
        X_train = pipe_image(X_train, pipe, 224)
        X_valid = pipe_image(X_valid, pipe, 224)
        X_test = pipe_image(X_test, pipe, 224)



    model.fit(X_train, y_train, validation_data = (X_valid, y_valid), shuffle = True)

    # Load best model in
    #model.model.load_weights(model_path)
    train_val_acc_loss = {
        'train_loss': model.model.history.history['loss'],
        'train_acc': model.model.history.history['sparse_categorical_accuracy'],
        'valid_loss' : model.model.history.history['val_loss'],
        'valid_acc' : model.model.history.history['val_sparse_categorical_accuracy']
    }

    y_pred = model.predict(X_test)
    cm = plot_cm(y_pred, y_test, fold_num, images_dir)
    performance_metrics = calculate_performance_metrics(y_test, y_pred, cm, fold_num)

    save_fold_results(epoch_counts, performance_metrics, fold_num, spreadsheet_file)
    plot_train_val_acc_loss2(train_val_acc_loss, fold_num, images_dir)

    performance_metrics_all.append(performance_metrics)
    fold_num += 1


# Average performance over all folds
performance_metrics_mean = {
    'Accuracy': np.mean([x['accuracy'] for x in performance_metrics_all]),
    'Precision': np.mean([x['precision'] for x in performance_metrics_all]),
    'Recall': np.mean([x['recall'] for x in performance_metrics_all]),
    'Sensitivity': np.mean([x['sensitivity'] for x in performance_metrics_all]),
    'Specificity': np.mean([x['specificity'] for x in performance_metrics_all]),
    'F1': np.mean([x['f1'] for x in performance_metrics_all])
}

now = datetime.now()
dt_end = now.strftime("%d-%m-%Y %H.%M.%S")
timestamps = {
    "start": dt_start,
    "end": dt_end
}
save_mean_results(performance_metrics_mean, timestamps, spreadsheet_file)
print(f"All testing completed with average test accuracy: {performance_metrics_mean['Accuracy']}\nStart: {dt_start} End: {dt_end}")
# %%


# train_acc = best_model.model.history.history['sparse_categorical_accuracy']
# train_loss = best_model.model.history.history['loss']
# val_acc = best_model.model.history.history['val_sparse_categorical_accuracy']
# val_loss = best_model.model.history.history['val_loss']

# plot_train_val_acc_loss(val_acc, val_loss, train_acc, train_loss, images_dir)


# y_pred = best_model.predict(X_test)
# results = {}
# results['y_pred'] = y_pred # Convert softmax output to 0 or 1
# results['y_test'] = y_test

# results_all.append(results)


# save_fold_info(i, hyp_results, performance_metrics, spreadsheet_file)

# performance_metrics_all.append(performance_metrics)

# plot_fold_test(y_pred, y_test, i, images_dir)




# %% hyperparam tuning
# model = KerasClassifier(build_fn = build_fn, batch_size = total_batch_size)
# search = GridSearchCV(estimator = model, param_grid = args['param_grid'], n_jobs = 1, refit = True, cv = cv_inner, scoring = 'accuracy') # cross-entropy loss
# search.fit(X_train, y_train, groups_train)    # Groups necessary only if inter-patinet used
# cv_results = search.cv_results_

# best_model = search.best_estimator_
# best_hyperparams = search.best_params_
# save_validation_results(cv_results, args['n_splits'], spreadsheet_file)


# # Retrain model using best hyperparameters found on trainval dataset
# best_model.fit(X_train, y_train, validation_data = (X_test, y_test))
# best_model.model.save(f'{models_dir}/Best model.h5')
# train_acc = best_model.model.history.history['sparse_categorical_accuracy']
# train_loss = best_model.model.history.history['loss']
# val_acc = best_model.model.history.history['val_sparse_categorical_accuracy']
# val_loss = best_model.model.history.history['val_loss']

# plot_train_val_acc_loss(val_acc, val_loss, train_acc, train_loss, images_dir)


# y_pred = best_model.predict(X_test)
# results = {}
# results['y_pred'] = y_pred # Convert softmax output to 0 or 1
# results['y_test'] = y_test
# # np.save(f'{test_pred_dir}/Fold {i} y_pred y_test.npy', results)
# # results_all.append(results)

# # Confusion matrix
# cm = plot_cm(y_pred, y_test, images_dir)
# # cm_all.append(cm)

# performance_metrics = calculate_performance_metrics(y_test, y_pred, cm)

# # save_fold_info(i, hyp_results, performance_metrics, spreadsheet_file)
# save_test_results(performance_metrics, best_hyperparams, spreadsheet_file)
# # performance_metrics_all.append(performance_metrics)

# # plot_fold_test(y_pred, y_test, i, images_dir)

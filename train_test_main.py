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
from tensorflow.keras.models import load_model
from train_test_helpers import *
from train_test_model import *
from train_test_parameters import *
import sys
import PIL
import re
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

from train_test_keras_helpers import *
import gc

np.random.seed(42)
# np.seterr(all='raise')
# np.seterr(all='warn')
# We need to split the dataset into sleep stages
X = []
y = []
groups = []
sample_PID = []
num_frames_ins = 0
num_frames_con = 0
num_patients_ins = 0
num_patients_con = 0
# Berlin. can shuffle if desired
#insomniaIDs = [1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75]
#goodIDs = [3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65]
#pIDs = ["ins2"]

allowed_stages = {
    "ALL": ["S1", "S2", "S3", "S4", "R"],
    "LSS": ["S1", "S2"],
    "N1": ["S1"],
    "N2": ["S2"],
    "SWS": ["S3", "S4"],
    "R": ["R"]
}
    
print("Loading patient data...")

# Determine which patients go in which groups
df_metadata = pd.read_csv(file_metadata)
group_dict, sample_PID_dict = group_greedy(df_metadata, pIDs, args['n_splits'])

if args["model"] == "AlexNet_1D":
    if args['subdataset'] == 'R':
        args['subdataset'] = 'REM'
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
        if args['balance_stage'] == True:
            threshold = max_num_epochs(stages)
            df = balance_dataset(df, threshold, args['subdataset'])   # Balance distribution of 5 types of subdataset
        df = get_sleep_epochs(df, args['subdataset'])
        if pClass == 'I':
            #num_frames_ins += len(df)
            num_patients_ins += 1
        else:
            #num_frames_con += len(df)
            num_patients_con += 1
        # Because threshold can be different size, we need to append epoch data one by one to list
        
        [X.append(row) for row in df.iloc[:, 1:None].to_numpy()]
        [y.append(pClass_dict[pClass]) for i in range(len(df.iloc[:, 0]))]
        [groups.append(group_dict[pID]) for i in range(len(df.iloc[:, 0]))]
        [sample_PID.append(pID) for i in range(len(df.iloc[:, 0]))]
elif args['model'] == "AlexNet_2D":
    seen_pIDs_ins = []
    seen_pIDs_con = []
    ORIGINAL_SIZE = (256, 256)
    PRECROPPED_SIZE = (256, 227)
    # PRECROPPED_SIZE = (224, 224)
    # CROPPED_SIZE = (227, 227)
    count = 0
    for file_image in os.listdir(path_tfr_folder):
        pID, frame, stage = file_image.split("-")
        stage = stage.split(".")[0] # remove png extension
        if pID in pIDs and stage in allowed_stages[args["subdataset"]]: # Ignore 'Other' and excluded stages from current test
            # print(file_image + " OK")
            img = Image.open(path_tfr_folder + file_image).convert('L')
            img = img.resize(PRECROPPED_SIZE)   # Resize vertical axis
            img = np.array(img)
            
            # img = img.reshape((img.shape[0], img.shape[1], 1))  # tensor
            pClass = df_metadata.loc[df_metadata["pID"] == pID, 'pClass'].values[0]
            if pClass == 'I':
                #num_frames_ins += 1
                if pID not in seen_pIDs_ins:
                    seen_pIDs_ins.append(pID)
                    num_patients_ins += 1
            elif pClass == 'G':
                #num_frames_con += 1
                if pID not in seen_pIDs_con:
                    seen_pIDs_con.append(pID)
                    num_patients_con += 1
            else:
                print("Error, incorrect pclass identified")
                break
            X.append(img)
            y.append(pClass_dict[pClass])
            groups.append(group_dict[pID])
            sample_PID.append([pID])
            count += 1
            # debug
            # if count == 10000:
            #     break
            if count % 1000 == 0:
                print(f"Loaded {count} images...")

assert(len(X) == len(y))
assert(len(y) == len(groups))     
print("All patient data loaded")
print("Conversion into np arrays")
mem_usage()
X = np.asarray(X)
mem_usage()
y = np.asarray(y)
mem_usage()
groups = np.asarray(groups)
mem_usage()
sample_PID = np.asarray(sample_PID)
mem_usage()

num_frames_con, num_frames_ins = np.bincount(y) 
# X, y, groups = custom_preprocess(X, y, groups)
# Apply class balancing
r = Reshape2Dto1D()
if args['model'] == 'AlexNet_2D':
    X = r.transform(X)
print(X[0], y[0], groups[0])
# if args['balance_class'] == True:
#     X, y, groups, sample_PID = class_balance(X, y, groups, sample_PID, args['n_splits'])

# Print statistics on dataset
# num_frames_con, num_frames_ins = np.bincount(y) 
print(f"Subdataset {args['subdataset']}: X.shape {X.shape} Y.shape {y.shape} group.shape {groups.shape} I frames:{num_frames_ins} G frames:{num_frames_con} I patients:{num_patients_ins} G patients:{num_patients_con}")


# Make directories for fold information, models, performance metrics, images, test and predicted data
now = datetime.now()
dt_start = now.strftime("%d-%m-%Y %H.%M.%S")
results_dir, models_dir, images_dir = create_dirs(args['title'], dt_start)

spreadsheet_file = f'./results/{args["spreadsheet"]}.xlsx'
spreadsheet_file2 = f'./results/{args["spreadsheet2"]}.xlsx'
spreadsheet_file_og = f'{results_dir}Result summary {dt_start}.xlsx'
with pd.ExcelWriter(f'{spreadsheet_file_og}') as writer:
    pass

# job_record_file = './results/Job record.xlsx'
strategy = tf.distribute.MirroredStrategy()
total_batch_size = args['batch_size'] * strategy.num_replicas_in_sync
print(f'Number of devices: {strategy.num_replicas_in_sync}\nBatch size per GPU: {args["batch_size"]}\nTotal batch size: {total_batch_size}')

performance_metrics_all = []
# performance_metrics_weighted = []
# fold_weights = []
results_all = []
cm_all = []

if args['model'] == 'AlexNet_1D':
    build_fn = create_model_AlexNet_1D
elif args['model'] == 'AlexNet_2D':
    build_fn = create_model_AlexNet_2D

if args['method'] == 'inter':
    cv_outer = GroupKFold(n_splits = args['n_splits'])
elif args['method'] == 'intra':
    cv_outer = KFold(n_splits = args['n_splits'], shuffle = True)

data_info = {
    'X.shape': str(X.shape),
    'y.shape': str(y.shape),
    'Insomnia': num_frames_ins,
    'Control': num_frames_con
}
# offset = save_parameters(args, data_info, spreadsheet_file_og, sheet_name)
# Create workbook
#  total_test_frames = y.shape[0]
# %%
# Split data into train and test sets

groups_used_for_valid = list(range(0, args["n_splits"]))  # Ensure unique validation set across all train folds

fold_num = 0
print("Pre-training memory")
mem_usage()
for i, (train_valid_index, test_index) in zip(range(100), cv_outer.split(X, y, groups)):
    print("Pre-split 1 memory")
    mem_usage()
    X_train_valid, y_train_valid, groups_train_valid, sample_PID_train_valid, X_test, y_test, groups_test, sample_PID_test = get_train_test2(X, y, groups, sample_PID, train_valid_index, test_index)
    print("Post-split 1 memory")
    mem_usage()
    unique_test_groups = np.unique(groups_test)
    unique_test_sample_PID = np.unique(sample_PID_test)
    # Further split train data into train and valid set
    if args['method'] == 'inter':
        cv_inner = GroupKFold(n_splits = args['n_splits'] - 1)
        
        valid_group_num = (unique_test_groups[0] + 1) % args['n_splits']
        # We also need a group to be distinct for validation set, for all folds

        # Get train indices for training group
        # train_index = np.array([x for i, x in enumerate(train_valid_index) if groups[i] != valid_group_num])
        train_index = []
        for j, group in enumerate(groups_train_valid):
            if group != valid_group_num:
                train_index.append(j)
        train_index = np.array(train_index)
        # Get valid indices for valid group
        valid_index = []
        for j, group in enumerate(groups_train_valid):
            if group == valid_group_num:
                valid_index.append(j)
        valid_index = np.array(valid_index)
    
    elif args['method'] == 'intra':
        cv_inner = KFold(n_splits = args['n_splits'] - 1, shuffle = True)
        train_index, valid_index = next(cv_inner.split(X_train_valid, y_train_valid, groups_train_valid))
    print("Pre-split 2 memory")
    mem_usage()
    X_train, y_train, groups_train, sample_PID_train, X_valid, y_valid, groups_valid, sample_PID_valid = get_train_test2(X_train_valid, y_train_valid, groups_train_valid, sample_PID_train_valid, train_index, valid_index)
    print("Post-split 2 memory")
    mem_usage()
    del X_train_valid
    del y_train_valid
    del groups_train_valid
    del sample_PID_train_valid
    print("Train valid variable del memory")
    mem_usage()
    # for train_index, valid_index in cv_inner.split(X_train_valid, y_train_valid, groups_train_valid):
    #     chosen_group = groups_train_valid[valid_index][0]
    #     if chosen_group in groups_used_for_valid:   # To get a unique validation set per train-test split
    #         groups_used_for_valid.remove(chosen_group)
    #         print(f"validation with group = {chosen_group}")
    #         print(f"remaining groups {groups_used_for_valid}")
    #         break

    if args['balance_class'] == True:
        X_train, y_train, groups_train = class_balance(X_train, y_train, groups_train, args['n_splits'])
    print('Post class balancing memory')
    mem_usage()
    unique_train_groups = np.unique(groups_train)   # Groups only needed for GroupKFold
    unique_valid_groups = np.unique(groups_valid)

    
    unique_train_sample_PID = np.unique(sample_PID_train)
    unique_valid_sample_PID = np.unique(sample_PID_valid)
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
        'Train PIDs': str(unique_train_sample_PID),
        'Valid groups': str(unique_valid_groups),
        'Valid PIDs': str(unique_valid_sample_PID),
        'Test groups': str(unique_test_groups),
        'Test PIDs': str(unique_test_sample_PID),
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
    num_test_frames = y_test.shape[0]
    # es = EarlyStopping(monitor = 'loss', mode = 'min', verbose=1, min_delta = )
    # best_model_info = "Epoch:{epoch:02d}\n{sparse_categorical_accuracy:.2f}-{val_sparse_categorical_accuracy:.2f}-{loss:.4f}-{val_loss:.4f}"

    model = KerasClassifier(build_fn = build_fn, epochs = args['epochs'], batch_size = total_batch_size, augment = args['augment'], std = args['std'], C = args['C'], lr = args['lr'], dropout_cnn = args['dropout_cnn'], dropout_dense = args['dropout_dense'])
    model2 = model
    if args['model'] == "AlexNet_1D":
        pipe_valid = Pipeline([
            ('scale1D', scale1D()),
            ('standardscaler', StandardScaler(with_mean = True, with_std = True)),
            ('reshapetotensor', ReshapeToTensor()),
        ])
    elif args['model'] == "AlexNet_2D":
        pipe_valid = Pipeline([
            ('scale2D', scale2D()),
            ('standardscaler', StandardScaler(with_mean = True, with_std = True)),   # Unit variance and zero mean
            ('reshapetotensor', ReshapeToTensor()),
            ('1dto2d', Reshape1Dto2D(height = 227, width = 256)),
        ])
    print("Pre-fit transform memory")
    mem_usage()
    X_train = pipe_valid.fit_transform(X_train)
    X_valid = pipe_valid.transform(X_valid)
    X_test = pipe_valid.transform(X_test)
    y_train = np.expand_dims(y_train, -1)
    y_valid = np.expand_dims(y_valid, -1)
    y_test = np.expand_dims(y_test, -1)
    
    # Save the best model only
    model_path=f"{results_dir}best_model.ckpt"
    mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min',  save_best_only=True, save_weights_only=True)
    

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), shuffle = True, callbacks = [mc], verbose = 2)


    
    # Load best model in
    train_val_acc_loss = {
        'train_loss': model.model.history.history['loss'],
        'train_acc': model.model.history.history['sparse_categorical_accuracy'],
        'valid_loss' : model.model.history.history['val_loss'],
        'valid_acc' : model.model.history.history['val_sparse_categorical_accuracy']
    }
    
    np.save(f"{results_dir}fold{i}_acc_loss.npy", train_val_acc_loss)
    # to load, do np.load(filename)
    # double check model was saved
    print(f"All valid loss: {model.model.history.history['val_loss']}\nall valid acc: {model.model.history.history['val_sparse_categorical_accuracy']}")
    # model2
    min_valid_loss = np.min(train_val_acc_loss['valid_loss'])
    index = np.argmin(train_val_acc_loss['valid_loss'])
    corr_valid_acc = train_val_acc_loss['valid_acc'][index]
    
    print(f"Min valid loss: {min_valid_loss} with valid acc {corr_valid_acc} at index {index}")
    print("Loading best model...")
    model.model.load_weights(model_path)
    # # model.model.load_weights(model_path)
    # print("Loaded best model and checking validation acc")
    valid_pred = model2.predict(X_valid) # 531,
    # y_valid has shape 531, 1
    print(f"Chosen best model valid accuracy: {accuracy_score(y_valid, valid_pred)}")


    # TEST
    # FRAME LEVEL CLASSIFICATION
    y_pred = model.predict(X_test)
    # print(y_pred, y_test)
    cm = plot_cm(y_pred, y_test, fold_num, images_dir)
    performance_metrics = calculate_performance_metrics(y_test, y_pred, cm, fold_num)
    try:
        save_fold_results(epoch_counts, performance_metrics, fold_num, spreadsheet_file_og, sheet_name)
    except Exception as e:
        print("Error saving")
        print(e)
    plot_train_val_acc_loss2(train_val_acc_loss, fold_num, images_dir)

    performance_metrics_all.append(performance_metrics)

    # PATIENT LEVEL CLASSIFICATION
    iteration_test_info = []
    patients_test = np.unique(sample_PID_test)

    for patient in patients_test:
        # Get frames belonging to test patient
        idxs_test = [idx for idx, pid in enumerate(sample_PID_test) if pid == patient]
        unique, counts = np.unique(y_pred[idxs_test], return_counts = True)
        patient_frame_classification = dict(zip(unique, counts))  # 0 = control, 1 = insomnia
        assert(np.max(y_test[idxs_test]) == np.min(y_test[idxs_test]))
        patient_class = y_test[idxs_test][0][0]
        # num_patient_frames_con, num_patient_frames_ins = np.bincount(y_pred[idxs_test]) 
        num_patient_frames_con = y_pred[idxs_test][y_pred[idxs_test] == 0].shape[0]
        num_patient_frames_ins = y_pred[idxs_test][y_pred[idxs_test] == 1].shape[0]
        # predicted_class = 0 if num_patient_frames_con > num_patient_frames_ins else 1
        # correct = 1 if predicted_class == patient_class else 0
        patient_frame_class = {
            'title': args['title'],
            'model': args['model'],
            'database': args['specific_dataset'],
            'sleep_stage': args['subdataset'],
            'iteration': i,
            'pID': patient,
            'con_frames': num_patient_frames_con,
            'ins_frames': num_patient_frames_ins,
            # 'predicted_pClass': predicted_class,
            'pClass': patient_class
            # 'correct': correct
        }
        iteration_test_info.append(patient_frame_class)
# %%
    save_fold_patient(i, spreadsheet_file2, "Results", iteration_test_info)

# %%
    # fold_weight = num_test_frames / total_test_frames
    # fold_weights.append(fold_weight)
        
    fold_num += 1
    # Delete saved model
    for fname in os.listdir(results_dir):
        if fname.startswith("best_model.ckpt"):
            # os.replace(f"{results_dir}{fname}", f"./models/{args['title']}{fname}")
            os.remove(os.path.join(results_dir, fname))

    del X_train
    del X_valid
    del X_test
    del y_train
    del y_valid
    del y_test
    
# Average performance over all folds
performance_metrics_mean = {
    'accuracy': np.mean([x['accuracy'] for x in performance_metrics_all]),
    'accuracy_std': np.std([x['accuracy'] for x in performance_metrics_all]),
    'precision': np.mean([x['precision'] for x in performance_metrics_all]),
    'precision_std': np.std([x['precision'] for x in performance_metrics_all]),
    'recall': np.mean([x['recall'] for x in performance_metrics_all]),
    'recall_std': np.std([x['recall'] for x in performance_metrics_all]),
    'sensitivity': np.mean([x['sensitivity'] for x in performance_metrics_all]),
    'sensitivity_std': np.std([x['sensitivity'] for x in performance_metrics_all]),
    'specificity': np.mean([x['specificity'] for x in performance_metrics_all]),
    'specificity_std': np.std([x['specificity'] for x in performance_metrics_all]),
    'f1': np.mean([x['f1'] for x in performance_metrics_all]),
    'f1_std': np.std([x['f1'] for x in performance_metrics_all])
}

now = datetime.now()
dt_end = now.strftime("%d-%m-%Y %H.%M.%S")
timestamps = {
    "start": dt_start,
    "end": dt_end
}
# save_mean_results(performance_metrics_mean, timestamps, spreadsheet_file, sheet_name)
append_summary(args, performance_metrics_mean, timestamps, args['number'], spreadsheet_file, "Summaries")






print(f"All testing completed with average test accuracy: {performance_metrics_mean['accuracy']}\nStart: {dt_start} End: {dt_end}")
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
# %%
# noise = RandomGaussianNoise1D(0.5, 0, 0.01)
# b = []
# for a in X_train:
#     b.append(noise.noise(a))
# plt.rcParams["figure.figsize"] = (20, 15)
# plt.subplot(2, 1, 1)
# plt.plot(X[0])
# plt.title("Before standardization and normalization")
# plt.subplot(2, 1, 2)
# plt.title("After standardization and normalization")
# plt.plot(X_train[0])
# # %%
# plt.subplot(5, 1, 1)
# plt.plot(X_train[0])
# plt.subplot(5, 2, 1)
# plt.plot(X_train_piped[0])
# %%
#X_out2 = X_out2.numpy()
# plt.rcParams["figure.figsize"] = (20, 15)
# i = 0
# plt.subplot(2, 1, 1)
# plt.plot(X_train[i])
# plt.subplot(2, 1, 2)
# plt.plot(X_out[-1])
# # %%
# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# train_ds2 = prepare(train_ds, batch_size, shuffle=False, augment=True)

# # %%
# for x in range(10):
#     out = flip(X_train[0])
#     plt.plot(out)
# %%
    # X_train, y_train, groups_train = custom_preprocess(X_train, y_train, groups_train)
    # X_valid, y_valid, groups_valid = custom_preprocess(X_valid, y_valid, groups_valid)
    # X_test, y_test, groups_test = custom_preprocess(X_test, y_test, groups_test)

    #model.fit(X_train, y_train, validation_data=(X_valid, y_valid), shuffle = True)

        


    # elif args["loading"] == "new":
    #     train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #     valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    #     test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    #     print(f"Augment is {True == args['augment']}")

    #     train_ds = prepare1D(train_ds, batch_size = args['batch_size'], std = args['std'], shuffle=True, augment=args['augment'])
    #     valid_ds = prepare1D(valid_ds, batch_size = args['batch_size'], std = args['std'], shuffle=False, augment=False)
    #     test_ds = prepare1D(test_ds, batch_size = args['batch_size'], std = args['std'], shuffle=False, augment=False)

    #     model = create_model_AlexNet_1D(C = args['C'], lr = args['lr'], std = args['std'], dropout_cnn = args['dropout_cnn'], dropout_dense = args['dropout_dense'])
    #     model.fit(
    #         train_ds, epochs = args['epochs'], validation_data = valid_ds
    #     )

    #model = create_model_AlexNet_1D()
    #model.fit(train_ds, validation_data=valid_ds, epochs=args['epochs'])

    # pipe = Pipeline([
    #     ('model', model)
    # ])

    # if args['augment'] == True and args['model'] == 'AlexNet_1D':
    #     pipe = Pipeline([
    #         ('randomflip1d', RandomFlip1D()),
    #         ('standardscaler', StandardScaler()),   # Standardization to zero mean, unit variance
    #         ('minmaxscaler', MinMaxScaler(feature_range = (-1, 1))), # Normalization


    #         ('reshapetotensor', ReshapeToTensor()),
    #         ('model', model)
    #     ])
    # elif args['augment'] == True and args['model'] == 'AlexNet_2D':
    #     pipe = Pipeline([
    #         ('standardscaler', StandardScaler()),   # Standardization to zero mean, unit variance
    #         ('minmaxscaler', MinMaxScaler(feature_range = (-1, 1))), # Normalization
    #         ('reshapetotensor', ReshapeToTensor()),
    #         ('model', model)
    #     ])
    # else:
    #     pipe = Pipeline([
    #         ('standardscaler', StandardScaler()),   # Standardization to zero mean, unit variance
    #         ('minmaxscaler', MinMaxScaler(feature_range = (-1, 1))), # Normalization
    #         ('reshapetotensor', ReshapeToTensor()),
    #         ('model', model)
    #     ])
    # We need to pipeline datasets individually, because validation data does not get pipelined
    # if args['model'] == "AlexNet_1D":
    #     X_train = pipe.fit_transform(X_train)
    #     X_valid = pipe.fit_transform(X_valid)
    #     X_test = pipe.fit_transform(X_test)

    # elif args['model'] == "AlexNet_2D":
    #     X_train = pipe_image(X_train, pipe, 227)
    #     X_valid = pipe_image(X_valid, pipe, 227)
    #     X_test = pipe_image(X_test, pipe, 227)
    #     # if args['augment'] == True:
    #     #     train_datagen = ImageDataGenerator(
    #     #         horizontal_flip=True,  # horizontal flip
    #     #     )  # brightness

    # X_train = pipe2.fit_transform(X_train)
    # X_valid = pipe2.fit_transform(X_valid)
    # model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = args['epochs'])
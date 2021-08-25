# %%

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from tensorflow.keras.utils import plot_model
import pydot_ng as pydot
from tensorflow.python.keras.backend import relu
import pickle
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut, GroupKFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
import gc
from datetime import datetime
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import seaborn as sn
import os
import json


def max_num_epochs(stages):
    max_epochs = 800
    num_ALL = stages['W'] + stages['S1'] + stages['S2'] + stages['S3'] + stages['S4'] + stages['R']
    num_LSS = stages['S1'] + stages['S2']
    num_SWS = stages['S3'] + stages['S4']
    num_REM = stages['R']
    num_BSL = stages['W'] + stages['S1'] + stages['S2'] + stages['R']
    threshold = min(max_epochs, num_ALL, num_LSS, num_SWS, num_REM, num_BSL)
    print(f"W {stages['W']} S1 {stages['S1']} S2 {stages['S2']} S3 {stages['S3']} S4 {stages['S4']} R {stages['R']}")
    print(f'ALL {num_ALL}, LSS {num_LSS}, SWS {num_SWS}, REM {num_REM}, BSL {num_BSL}\nChosen threshold {threshold}')
    return threshold

def create_model_AlexNet():
    initializer = tf.keras.initializers.HeNormal()  # Kaiming initializer
    AlexNet = keras.Sequential([
        # CNN 1
        keras.layers.Conv1D(filters = 64, kernel_size = 11, strides = 4,  padding = 'same', input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4), data_format = 'channels_last'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),

        # CNN 2
        keras.layers.Conv1D(filters = 192, kernel_size = 5, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4), data_format = 'channels_last'),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),

        # CNN 3
        keras.layers.Conv1D(filters = 384, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4), data_format = 'channels_last'),
        keras.layers.Activation('relu'),

        # CNN 4
        keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4), data_format = 'channels_last'),
        keras.layers.Activation('relu'),

        # CNN 5
        keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4), data_format = 'channels_last'),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),
        keras.layers.AveragePooling1D(pool_size = 18, strides = 18, padding = 'valid', data_format = 'channels_last'),

        # Flatten
        keras.layers.Flatten(),

        # 3 Fully Connected Layers
        keras.layers.Dense(units = 512, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(units = 128,  kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(units = 2, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('softmax') # Output activation is softmax
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    AlexNet.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy'])
    return AlexNet

def balance_dataset(df, threshold, subdataset):
    if subdataset == 'ALL':
        data = df
    elif subdataset == 'LSS':
        data = df.loc[(df['Sleep_Stage'] == 'S1') |  (df['Sleep_Stage'] == 'S2')]
    elif subdataset == 'SWS':
        data = df.loc[(df['Sleep_Stage'] == 'S3') |  (df['Sleep_Stage'] == 'S4')]
    elif subdataset == 'REM':
        data = df.loc[(df['Sleep_Stage'] == 'R')]
    elif subdataset == 'BSL':
        data = df.loc[(df['Sleep_Stage'] == 'W') |  (df['Sleep_Stage'] == 'S1') | (df['Sleep_Stage'] == 'S2') | (df['Sleep_Stage'] == 'R')]
    data_resampled = resample(data, replace = False, n_samples = threshold, random_state = 42)
    count = data_resampled['Sleep_Stage'].value_counts()
    print(f'After balancing:\n{count}')
    return data_resampled

class ReshapeToTensor():
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X = np.expand_dims(X, -1)
        #y = np.expand_dims(y, -1)
        return X
# A hack to plot confusion matrix with keras model
class MyModelPredict(object):
    def __init__(self, model):
        self._estimator_type = 'classifier'
        self.model = model
        
    def predict(self, X_test):
        m = self.model
        y_pred = m.predict_classes(X_test)
        return y_pred


### CHANGE THESE
dataset = 'CAP_Overlap'
balance = True
subdataset = 'SWS'
results_dir = './CAP results/reference/'
###

dataPath = f'F:\\Sleep data formatted\\{dataset}.h5'
numEpochDataPoints = 128*30
if dataset == 'CAP_Overlap':
    pIDs = ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n1', 'n2', 'n3', 'n4', 'n5', 'n10', 'n11', 'n12', 'n14']
    # For GroupKFold CV
    group_dict = {'n1': 1, 'n2': 10, 'n3': 10, 'n4': 4, 'n5': 5, 'n10': 6, 'n11': 7, 'n12': 8, 'n14': 9,
        'ins1': 1, 'ins2': 10, 'ins3': 10, 'ins4': 4, 'ins5': 5, 'ins6': 6, 'ins7': 7, 'ins8': 8, 'ins9': 9}
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
# Berlin. can shuffle if desired
#insomniaIDs = [1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75]
#goodIDs = [3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65]

pClass_dict = {'G': 0, 'I': 1}

for pID in pIDs:
    pID = str(pID)
    print(f'pID: {pID}')
    # To avoid loading all data into RAM, we load only one patient at a time
    with pd.HDFStore(dataPath) as store:
        df = store[pID]
        pID, pClass, startTime, endTime, stages = (store.get_storer(pID).attrs.metadata)['overlap']

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
                group_dict[pID] = ins_patients
            else:
                num_control += len(df)
                con_patients += 1
                group_dict[pID] = con_patients
            # Because threshold can be different size, we need to append epoch data one by one to list
            [X.append(row) for row in df.iloc[:, 1:None].to_numpy()]
            [y.append(pClass_dict[pClass]) for i in range(len(df.iloc[:, 0]))]
            [groups.append(group_dict[pID]) for i in range(len(df.iloc[:, 0]))]

            
        


X = np.asarray(X)   # Use asarray to avoid making copies of array
#X = X.reshape(X.shape[0], X.shape[1], 1)
y = np.asarray(y)       
#y = y.reshape(y.shape[0], 1)  
groups = np.asarray(groups)
print(f'Subdataset {subdataset}: X.shape {X.shape} Y.shape {y.shape} group.shape {groups.shape} I {num_insomnia} G {num_control} ')

# K-fold into train and test datasets
gkf = GroupKFold(n_splits = n_splits)
# %%
#### %%capture cap --no-stderr

# Make directories for models, performance metrics, images, test and predicted data
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
models_dir = f'{results_dir}/models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
performance_dir = f'{results_dir}/performance_metrics'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)
images_dir = f'{results_dir}/images'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
test_pred_dir = f'{results_dir}/test_pred_npy'
if not os.path.exists(test_pred_dir):
    os.makedirs(test_pred_dir)

with open(f'{results_dir}test_info.txt', 'w') as f:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
    f.write(f'Start time: {dt_string}\nDataset: {dataset}\nSubdataset: {subdataset}\nFurther info: scaled to 0-1 range')

performance_metrics_all = []
results_all = []
cm_all = []

for (train_index, test_index), i in zip(gkf.split(X, y, groups), range(100)):
    #keras.backend.image_data_format()
    #keras.backend.set_image_data_format('channels_first')
    
    X_train = X[train_index].astype('float32')
    y_train = y[train_index]
    X_test = X[test_index].astype('float32')
    y_test = y[test_index]
    train_groups = np.unique(groups[train_index])
    test_groups = np.unique(groups[test_index])

    unique, counts = np.unique(y_train, return_counts = True)
    train_info = dict(zip(unique, counts))
    unique, counts = np.unique(y_test, return_counts = True)
    test_info = dict(zip(unique, counts))
    iteration = f'Train groups {train_groups}; test groups {test_groups}'
    info = f'{iteration}\nX_train {X[train_index].shape} y_train {y[train_index].shape} X_test {X[test_index].shape} y_test {y[test_index].shape}\nTrain info: {train_info} Test info: {test_info}'
    print(info)
    train_test_name = f'Train {train_groups} Test {test_groups}'


    AlexNet = KerasClassifier(build_fn = create_model_AlexNet, epochs = 80, batch_size = 256)
    pipe = Pipeline([
        #('standardscaler', StandardScaler()),   # Scale all points in each epoch to zero mean, and feature for all epochs to have unit variance (sum to occurrences)
        ('minmaxscaler', MinMaxScaler(feature_range = (0, 1))),
        ('reshape_to_tensor', ReshapeToTensor()),
        ('AlexNet', AlexNet)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Get and save performance metrics
    
    model = pipe[-1]
    model.model.save(f'{models_dir}{train_test_name} AlexNet.h5')

    ### Load test data
    # a = np.load(f'{folder}y_pred y_test.npy', allow_pickle = True)
    # y_pred = a.item()['y_pred']
    # y_test = a.item()['y_test']
    ###
    results = {}
    results['y_pred'] = y_pred
    results['y_test'] = y_test
    np.save(f'{results_dir}{train_test_name} y_pred y_test.npy', results)
    results_all.append(results)

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Control (0)', 'Insomnia (1)']).plot(cmap = 'Blues')
    plt.title(f'Fold {i+1} confusion matrix')
    plt.savefig(f'{images_dir}{train_test_name} CM.png', dpi = 100)
    plt.clf()
    cm_all.append(cm)

    cm_norm = confusion_matrix(y_test, y_pred, normalize = 'true')
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_norm, display_labels = ['Control (0)', 'Insomnia (1)']).plot(cmap = 'Blues')
    plt.title(f'{train_test_name} Confusion matrix')
    plt.savefig(f'{images_dir}{train_test_name} CM normalized.png', dpi = 100)
    plt.clf()

    performance_metrics = {}
    performance_metrics['accuracy'] = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    performance_metrics['precision'] = tp/(tp + fp)
    performance_metrics['recall'] = tp/(tp + fn)
    performance_metrics['sensitivity'] = tp/(tp + fn)
    performance_metrics['specificity'] = tn/(tn + fp)
    performance_metrics['f1'] = 2 * performance_metrics['precision'] * performance_metrics['recall'] / (performance_metrics['precision'] + performance_metrics['recall'])
    np.save(f'{performance_dir}{train_test_name} performance_metrics.npy', performance_metrics)
    plt.clf()
    performance_metrics_all.append(performance_metrics)
    
    with open(f'{performance_dir}{train_test_name} performance_metrics.txt', 'w') as f:
        f.write(info)
        f.write(json.dumps(performance_metrics))
        

    plt.plot(y_pred, label = 'pred', linestyle = 'None', markersize = 1.0, marker = '.')
    plt.plot(y_test, label = 'test')
    plt.title(f'{train_test_name} Test vs predicted')
    plt.ylabel('Control (0), Insomnia (1)')
    plt.xlabel('Test epoch')
    plt.legend()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.savefig(f'{images_dir}{train_test_name} Test vs predicted.png', dpi = 200)
    plt.clf()

    print(f"Fold {i} Acc: {performance_metrics['accuracy']}")

performance_metrics_avg = [
    np.mean([x['accuracy'] for x in performance_metrics_all]),
    np.mean([x['precision'] for x in performance_metrics_all]),
    np.mean([x['recall'] for x in performance_metrics_all]),
    np.mean([x['sensitivity'] for x in performance_metrics_all]),
    np.mean([x['specificity'] for x in performance_metrics_all]),
    np.mean([x['f1'] for x in performance_metrics_all])
]

np.save(f'{results_dir}performance_metrics_avg.npy', performance_metrics_avg)
with open(f'{results_dir}performance_metrics_avg.txt', 'w') as f:
    f.write(json.dumps(performance_metrics_avg))
# %%
with open(f'{results_dir}timestamp.txt', 'a') as f:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
    f.write(f'\nEnd time: {dt_string}')
# %%
# JUPYTER OUTPUT
with open(f'{results_dir}ipython_output.txt', 'w') as f:
    f.write(cap.stdout)
# %%
#model = keras.models.load_model(os.path.join('./CAP results/balanced data epoch 80 rem models/', f))
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
for m, x in zip(models, range(1, 10)):
    m[0].model.save(f'./CAP results/balanced data epoch 80/{dt_string} fold_{x}.h5')
results = np.array(results)
np.save('./CAP results/overlap epoch 20/results', results)
# %%
import os
#for f in os.listdir('./CAP results/balanced data epoch 80/'):
    
model = keras.models.load_model(f'./CAP results/balanced data epoch 80/28-07-2021 18.45.41 fold_1.h5')
y_pred = model.predict(X_test)
# %%
    unique, counts = np.unique(y_test, return_counts=True)
    d = dict(zip(unique, counts))
    results.append((accuracy_score(y_test, y_pred), d))
    print(results)
    models.append(pipeline)
# %%

res = tf.math.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(models[8][0], X_test, y_test,
                             display_labels={"Control", "Insomnia"},
                             cmap=plt.cm.Blues,
                             normalize= True)
# %%
maxTrainPatients = 8
maxTestPatients = 1
numTrainInsomniaPatients = 0
numTrainGoodPatients = 0
numTestInsomniaPatients = 0
numTestGoodPatients = 0
numGoodTrainEpochs = 0
numGoodTestEpochs = 0
numInsomniaTrainEpochs = 0
numInsomniaTestEpochs = 0

X_train = []
X_test = []
Y_train = []
Y_test = []



# Code for CAP 
# insomniaIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# goodIDs = [11, 12, 13, 14, 15, 22, 24, 25, 26]
# IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 22, 24, 25, 26]
# Using inter-patient paradigm (split patients into training and testing set)
## %%
classDict = {'G': 0, 'I': 1}
df = pd.DataFrame(columns = ['All', 'SLEEP-S0', 'SLEEP-S1', 'SLEEP-S2', 'SLEEP-S3', 'SLEEP-S4', 'SLEEP-REM'], index = ['Control', 'Insomnia', 'Total'])
for col in df.columns:
    df[col].values[:] = 0

good_id_to_group = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 22:6, 24:7, 25:8, 26: 9}
data = []
groups = np.array([])
for i in goodIDs:
    f = pd.read_hdf(dataPath, key = str(i))
    for epoch in f:
        data.append([epoch[0], epoch[1], epoch[2], classDict[epoch[3]]])
        groups = np.append(groups, good_id_to_group[epoch[0]])
        df.at['Control', epoch[1]] += 1
        df.at['Total', epoch[1]] += 1
        df.at['Control', 'All'] += 1
        df.at['Total', 'All'] += 1
for i in insomniaIDs:
    f = pd.read_hdf(dataPath, key = str(i))
    for epoch in f:
        data.append([epoch[0], epoch[1], epoch[2], classDict[epoch[3]]])
        groups = np.append(groups, epoch[0])
        df.at['Insomnia', epoch[1]] += 1
        df.at['Total', epoch[1]] += 1
        df.at['Insomnia', 'All'] += 1
        df.at['Total', 'All'] += 1
        
data = np.asarray(data, dtype = object)

X = data[:, 2]
Y = data[:, 3]



lpgo = LeavePGroupsOut(n_groups=1)
lpgo.get_n_splits(X, Y, groups)

lpgo.get_n_splits(groups=groups)  # 'groups' is always required

print(lpgo)

for train_index, test_index in lpgo.split(X, Y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #print(X_train, X_test, y_train, y_test)

X_train = X_train.tolist()
X_train = np.reshape(X_train, (len(X_train), numEpochDataPoints, 1)).astype('float32')
Y_train = np.asarray(Y_train).astype('uint8')
Y_train = np.reshape(Y_train, (len(Y_train), 1))

X_test = X_test.tolist()
X_test = np.reshape(X_test, (len(X_test), numEpochDataPoints, 1)).astype('float32')
Y_test = np.asarray(Y_test).astype('uint8')
Y_test = np.reshape(Y_test, (len(Y_test), 1))
# %%

train = True
for g in goodIDs:
    print(g)
    data = pd.read_hdf(dataPath, key = str(g))
    if train == True:
        numTrainGoodPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_train.append(epoch[2])
            Y_train.append(epoch[3])
            numGoodTrainEpochs += 1
        if numTrainGoodPatients == maxTrainPatients:
            train = False
    else:
        numTestGoodPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_test.append(epoch[2])
            Y_test.append(epoch[3])
            numGoodTestEpochs += 1
        if numTestGoodPatients == maxTestPatients:
            break

train = True
for i in insomniaIDs:
    print(i)
    data = pd.read_hdf(dataPath, key = str(i))
    if train == True:
        numTrainInsomniaPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_train.append(epoch[2])
            Y_train.append(epoch[3])
            numInsomniaTrainEpochs += 1
        if numTrainInsomniaPatients == maxTrainPatients:
            train = False
    else:
        numTestInsomniaPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_test.append(epoch[2])
            Y_test.append(epoch[3])
            numInsomniaTestEpochs += 1
        if numTestInsomniaPatients == maxTestPatients:
            break


print("Epochs, Good train: {} | In train: {} | Good test: {} | In test: {}".format(numGoodTrainEpochs, numInsomniaTrainEpochs, numGoodTestEpochs, numInsomniaTestEpochs))

X_train = np.asarray(X_train).astype('float32')
X_train = np.reshape(X_train, (len(X_train), numEpochDataPoints, 1))
Y_train = [0 if x == 'G' else 1 for x in Y_train]
Y_train = np.asarray(Y_train).astype('uint8')
Y_train = np.reshape(Y_train, (len(Y_train), 1))

X_test = np.asarray(X_test).astype('float32')
X_test = np.reshape(X_test, (len(X_test), numEpochDataPoints, 1))
Y_test = [0 if x == 'G' else 1 for x in Y_test]
Y_test = np.asarray(Y_test).astype('uint8')
Y_test = np.reshape(Y_test, (len(Y_test), 1))

# %%
# Alexnet model instantializer

# %%
m = 31   # model number
  # Sparse weights make algorithm faster
# %%
history = model.fit(X_train, Y_train, batch_size = 256, epochs = 50, shuffle = True)
np.save('{} history.npy'.format(m),history.history)
# %%
model.evaluate(X_test, Y_test)
# %%
y_prob = model.predict(X_test) 
y_classes = y_prob.argmax(axis=-1)
# %%
# Confusion matrix
cm = confusion_matrix(y_true = Y_test ,y_pred=y_classes)
plot_confusion_matrix(cm = cm, classes = ['0 (control)', '1 (insomnia)'], title = "confusion matrix")
 # %%
plt.rcParams["figure.figsize"] = (10,5)
plt.plot(history.history['sparse_categorical_accuracy'], label = 'accuracy', color = 'g')
plt.plot(history.history['loss'], label = 'loss', color = 'r')
plt.title('Training accuracy/loss')
plt.ylabel('Accuracy/loss')
plt.xlabel('Training iteration')
plt.yticks(np.arange(0, 1, 0.05))
plt.grid()
plt.legend(bbox_to_anchor = (1, 1))
plt.savefig('model{}_accloss.png'.format(m), dpi = 200)
# %%
inter_output_model = keras.Model(model.input, model.get_layer(index = 23).output )
inter_output = inter_output_model.predict(X_test)
plt.rcParams["figure.figsize"] = (10,5)
plt.title('Predicted vs actual output for test data')
#plt.plot(inter_output[:, 0], label = 'Class 0 (Control)')
#plt.plot(inter_output[:, 1], label = 'Class 1 (Insomnia)')
plt.plot(y_classes, label = 'Predicted label')
plt.plot(Y_test, label = 'True label')
#plt.ylim([-2, 2])
plt.legend()
plt.xlabel('Test epoch #')
plt.ylabel('Softmax output')
plt.tight_layout()
plt.grid()

plt.savefig('model{}_output.png'.format(m), dpi = 200)
# %%
model.save("E:\\HDD documents\\University\\Thesis\\Thesis B code\\model{}.h5".format(m))
# %%
# loading stuff
n = -1
model = keras.models.load_model('E:\\HDD documents\\University\\Thesis\\Thesis B code\\model{}.h5'.format(n))
# %%
history=np.load('{} history.npy'.format(n),allow_pickle='TRUE').item()


# %%
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# %%

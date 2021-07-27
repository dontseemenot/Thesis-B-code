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
from keras import backend as K
import pickle
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut, GroupKFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
import gc
# %%
def create_model_AlexNet():
    initializer = tf.keras.initializers.HeNormal()  # Kaiming initializer
    AlexNet = keras.Sequential([
        # CNN 1
        keras.layers.Conv1D(filters = 64, kernel_size = 11, strides = 4,  padding = 'same', input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid'),

        # CNN 2
        keras.layers.Conv1D(filters = 192, kernel_size = 5, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid'),

        # CNN 3
        keras.layers.Conv1D(filters = 384, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('relu'),

        # CNN 4
        keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('relu'),

        # CNN 5
        keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(1e-4)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid'),
        keras.layers.AveragePooling1D(pool_size = 18, strides = 18, padding = 'valid'),

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
    
    #model.summary()
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    AlexNet.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy'])

    return AlexNet
# %%
datasetName = 'CAP_overlap'
#datasetName = 'Berlin'
if datasetName == 'CAP_overlap':
    # With overlap, scaled
    dataPath = 'F:\\Sleep data formatted\\CAP.h5'
    Fs = 128
    epochLength = 30
    numEpochDataPoints = Fs * epochLength
    
else:
    dataPath = 'F:\\Sleep data formatted\\alldataNormDown2.h5'
    Fs1 = 512
    Fs2 = 128
    epochLength = 30
    numEpochDataPoints = Fs2 * epochLength
# We need to split the dataset into sleep stages
df = pd.read_hdf(dataPath, key = datasetName)
# %%
# Balance dataset so num of different types of sleep stages are equal within the patient
threshold_per_pid = []
for pID, pClass in pd.unique(df['pID'], df['pClass']):
    num_all = 0
    num_LSS = 0
    num_SWS = 0
    num_REM = 0
    max_threshold = 800
    print(df.loc[df['pID'] == pID, 'Sleep_Stage'].value_counts())
    stages = df.loc[df['pID'] == pID, 'Sleep_Stage'].value_counts().index.tolist()
    for s in stages:
        num_all += df.loc[df['pID'] == pID, 'Sleep_Stage'].value_counts()[s]
        if s == 'S1' or s == 'S2':
            num_LSS += df.loc[df['pID'] == pID, 'Sleep_Stage'].value_counts()[s]
        elif s == 'S3' or s == 'S4':
            num_SWS += df.loc[df['pID'] == pID, 'Sleep_Stage'].value_counts()[s]
        elif s == 'REM':
            num_REM += df.loc[df['pID'] == pID, 'Sleep_Stage'].value_counts()[s]
    threshold = min(num_all, num_LSS, num_SWS, num_REM, max_threshold)
    print(f'pID {pID}\nAll {num_all}, LSS {num_LSS}, SWS {num_SWS}, REM {num_REM}\nChosen threshold {threshold}\n')
    threshold_per_pid.append((pID, threshold))        
    

# %%
group_dict = {'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4, 'n5': 5, 'n12': 6, 'n14': 7, 'n15': 8, 'n16': 9,
        'ins1': 1, 'ins2': 2, 'ins3': 3, 'ins4': 4, 'ins5': 5, 'ins6': 6, 'ins7': 7, 'ins8': 8, 'ins9': 9}
pClass_dict = {'G': 0, 'I': 1}
lpgo = LeavePGroupsOut(n_groups = 1)
X = np.array(df.iloc[:, 2:-1])
y = np.array([pClass_dict[pClass] for pClass in df.iloc[:, -1]])

groups = np.array([group_dict[stage] for stage in df.iloc[:, 0]])
print(f'X.shape {X.shape} Y.shape {y.shape} group.shape {groups.shape}')
del df
gc.collect()    # Free up memory used by df
# %%
# Inter-patient K-fold CV, K = 9
lpgo = LeavePGroupsOut(n_groups = 1)
lpgo.get_n_splits(X, y, groups)
for train_index, test_index, in lpgo.split(X, y, groups):    # returns generators
    X_train = X[train_index].reshape(len(X[train_index]), numEpochDataPoints, 1)
    y_train = y[train_index].reshape(len(y[train_index]), 1)
    X_test = X[test_index].reshape(len(X[test_index]), numEpochDataPoints, 1)
    y_test = y[test_index].reshape(len(y[test_index]), 1)
    print(f'X_train {X[train_index].shape} y_train {y[train_index].shape} X_test {X[test_index].shape} y_test {y[test_index].shape}')
# %%
AlexNet = KerasClassifier(build_fn = create_model_AlexNet, epochs = 10, batch_size = 256)
pipeline = Pipeline([
    ('AlexNet', AlexNet)
])

pipeline.fit(X_train, y_train)
# %%
y_pred = pipeline.predict(X_test)

# %%

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

# Berlin. can shuffle if desired
#insomniaIDs = [1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75]
#goodIDs = [3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65]

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

# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers


dataPath = 'F:\\Berlin data formatted\\alldata.h5'
Fs1 = 512
Fs2 = 128
epochLength = 30
numEpochDataPoints = Fs1 * epochLength

maxTrainPatients = 4
maxTestPatients = 1
numTrainInsomniaPatients = 0
numTrainGoodPatients = 0
numTestInsomniaPatients = 0
numTestGoodPatients = 0


X_train = []
X_test = []
Y_train = []
Y_test = []

# can shuffle if desired
goodIDs = [1, 2, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 26, 27, 41, 42, 43, 52, 53, 54, 55, 56, 57, 60, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 75]
insomniaIDs = [3, 7, 8, 9, 10, 11, 12, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 59, 61, 65]
# %%
# Load dataset
f = h5py.File(dataPath, "r")
# %%
# Using inter-patient paradigm (split patients into training and testing set)

train = True
for g in goodIDs:
    print(g)
    data = pd.read_hdf(dataPath, key = str(g))
    if train == True:
        numTrainGoodPatients += 1
        for epoch in data:
            X_train.append(epoch[2])
            Y_train.append(epoch[3])
        if numTrainGoodPatients == maxTrainPatients:
            train = False
    else:
        numTestGoodPatients += 1
        for epoch in data:
            X_test.append(epoch[2])
            Y_test.append(epoch[3])
        if numTestGoodPatients == maxTestPatients:
            break

train = True
for i in insomniaIDs:
    print(i)
    if train == True:
        data = pd.read_hdf(dataPath, key = str(i))
        numTrainInsomniaPatients += 1
        for epoch in data:
            X_train.append(epoch[2])
            Y_train.append(epoch[2])
        if numTrainInsomniaPatients == maxTrainPatients:
            train = False
    else:
        numTestInsomniaPatients += 1
        for epoch in data:
            X_test.append(epoch[2])
            Y_test.append(epoch[3])
        if numTestInsomniaPatients == maxTestPatients:
            break


# %%
for p in range(1, 76):  # patient num 1-75
    try:
        data = pd.read_hdf(dataPath, key = str(p))    # Use patient ID to index dataset
    except:
        print('Patient {} does not exist'.format(p))
#After you are done
f.close()
# %%
model = keras.Sequential([
    # CNN 1
    keras.layers.Conv1D(filters = 64, kernel_size = 11, strides = 4, activation = 'relu',  padding = 'same', input_shape = (numEpochDataPoints, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid'),

    # CNN 2
    keras.layers.Conv1D(filters = 192, kernel_size = 5, strides = 1, activation = 'relu',  padding = 'same'),
    keras.layers.ReLU(),
    keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid'),

    # CNN 3
    keras.layers.Conv1D(filters = 384, kernel_size = 3, strides = 1, activation = 'relu',  padding = 'same'),
    keras.layers.ReLU(),

    # CNN 4
    keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1, activation = 'relu',  padding = 'same'),
    keras.layers.ReLU(),

    # CNN 5
    keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1, activation = 'relu',  padding = 'same'),
    keras.layers.ReLU(),
    keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid'),
    keras.layers.AveragePooling1D(pool_size = 18, strides = 18, padding = 'valid'),

    # Flatten
    keras.layers.Flatten(),

    # 3 Fully Connected Layers
    keras.layers.Dense(units = 512, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units = 128, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units = 2, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4))

])
model.summary()
# %%
optimizer = keras.optimizers.Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(),  metrics = ['accuracy'])  # Sparse weights make algorithm faster
model.fit(batch_size = 256)

# %%
# LOOCV (do later)
# Q2f
lambdas2 = np.linspace(0.1, 20, int(20/0.1)) # Ignore lambda = 0 to avoid warnings
lambdaMSEsLasso = []
for l in range(len(lambdas2)):
    MSEsLasso = []
    for ni in range(np.shape(dataX)[0]):
        X_test = dataX[ni]
        X_train = dataX
        X_train = np.delete(X_train, ni, axis = 0)
        y_test = dataY[ni]
        y_train = dataY
        y_train = np.delete(y_train, ni, axis = 0)
        
        # Lasso model
        lasso = Lasso(alpha = lambdas2[l])
        lasso.fit(X_train, y_train)
        predYLasso = lasso.predict([X_test])
        MSEsLasso.append(mean_squared_error([y_test], predYLasso))

    lambdaMSEsLasso.append([lambdas[l], mean(MSEsLasso)])

    ax = plt.gca()

MSE = [el[1] for el in lambdaMSEsLasso]
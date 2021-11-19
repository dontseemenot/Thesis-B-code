import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, AveragePooling1D, Flatten, Dense, Dropout, GaussianNoise, Conv2D, MaxPool2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy



from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing
from train_test_keras_helpers import *

# 5 CNN + 3 Dense layers
def create_model_AlexNet_1D(augment, std, C, lr, dropout_cnn, dropout_dense):  # Default hyperparameters
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        initializer = HeNormal()  # Kaiming initializer
        numEpochDataPoints = 128*30
        AlexNet = Sequential([
            Conv1D(filters = 64, kernel_size = 11, strides = 4,  padding = 'same', input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),
            #Dropout(dropout_cnn),

            # CNN 2
            Conv1D(filters = 192, kernel_size = 5, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            Activation('relu'),
            MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),
            #Dropout(dropout_cnn),

            # CNN 3
            Conv1D(filters = 384, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            Activation('relu'),
            #Dropout(dropout_cnn),

            # CNN 4
            Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            Activation('relu'),
            #Dropout(dropout_cnn),

            # CNN 5
            Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            Activation('relu'),
            MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),
            AveragePooling1D(pool_size = 18, strides = 18, padding = 'valid', data_format = 'channels_last'),
            #Dropout(dropout_cnn),
            Flatten(),
            
            # Dense 1
            #Dropout(0.5),
            Dense(units = 512, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            Activation('relu'),
            Dropout(dropout_dense),

            # Dense 2
            Dense(units = 128,  kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            Activation('relu'),
            Dropout(dropout_dense),

            # Dense 3
            Dense(units = 2, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            Activation('softmax') # Output activation is softmax
        ])
        print(AlexNet.summary())
        if augment != "None":
            if augment == "All":
                data_aug = Sequential([
                    RandomFlip1D(input_shape = (numEpochDataPoints, 1)),
                    RandomGaussianNoise1D(std=std, input_shape = (numEpochDataPoints, 1))
                ])
            elif augment == "Flip":
                data_aug = Sequential([
                    RandomFlip1D(input_shape = (numEpochDataPoints, 1))
                ])
            elif augment == "Gaussian":
                data_aug = Sequential([
                    RandomGaussianNoise1D(std=std, input_shape = (numEpochDataPoints, 1))
                ])            
            print(data_aug.summary())
            AlexNet = Sequential([data_aug, AlexNet])
    optimizer = Adam(learning_rate = lr, epsilon = 1e-8)
    lr_metric = get_lr_metric(optimizer)
    AlexNet.compile(optimizer = optimizer, loss = SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy', lr_metric])
    # print(AlexNet.summary())
    return AlexNet

def create_model_AlexNet_2D(augment, std, C, lr, dropout_cnn, dropout_dense):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # initializer = tf.keras.initializers.HeNormal()  # Kaiming initializer
        input_shape = (227, 256, 1)
        cropped_shape = (227, 227, 1)
        AlexNet = Sequential([
            Resizing(cropped_shape[0], cropped_shape[1], input_shape=input_shape),    # Resize to 227 x 227, regardless if augmentation was performed or not
            Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=cropped_shape, padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same"),

            Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same"),

            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),

            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            BatchNormalization(),

            MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same"),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(dropout_dense),
            Dense(1000, activation='relu'),
            Dropout(dropout_dense),
            Dense(2, activation='softmax')
        ])
        print(AlexNet.summary())
        if augment != "None":
            if augment == "All":
                data_aug = Sequential([
                    RandomFlip2D(input_shape = input_shape),
                    RandomCrop2D(input_shape = input_shape)
                ])
            elif augment == "Flip":
                data_aug = Sequential([
                    RandomFlip2D(input_shape = input_shape)
                ])
            elif augment == "Crop":
                data_aug = Sequential([
                    RandomCrop2D(input_shape = input_shape)
                ])
            print(data_aug.summary())
            AlexNet = Sequential([data_aug, AlexNet])
    optimizer = Adam(learning_rate = lr, epsilon = 1e-8)
    lr_metric = get_lr_metric(optimizer)    # Note that this only returns upper bound, as adam optimizer will automatically adjust lr between 0 and upper bound.
    AlexNet.compile(optimizer = optimizer, loss = SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy', lr_metric])
    return AlexNet


def create_model_LeNet_549_1D(C = 0.0001, lr = 0.0001, std = 0.01):
    numEpochDataPoints = 128*30
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        initializer = HeNormal()  # Kaiming initializer
        LeNet = Sequential([
            GaussianNoise(stddev = std, input_shape = (numEpochDataPoints, 1)),
            Conv1D(filters = 6, kernel_size = 549, input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),
            AveragePooling1D(pool_size = 2, strides = 2),

            Conv1D(filters = 16, kernel_size = 549, kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),
            AveragePooling1D(pool_size = 2, strides = 2),

            Conv1D(filters = 120, kernel_size = 549, kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),

            # Flatten(),
            Dense(units = 60, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),
            Dense(units = 2, kernel_regularizer = regularizers.l2(C)),
            Activation('softmax')



            # # CNN
            # Conv1D(filters = 6, kernel_size = 52, input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            # Activation('tanh'),
            # AveragePooling1D(pool_size = 4, strides = 4),


            # # CNN 2
            # Conv1D(filters = 16, kernel_size = 52, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), kernel_initializer=initializer),
            # Activation('tanh'),
            # AveragePooling1D(pool_size = 4, strides = 4),

            # Flatten(),
            # Dense(units = 120, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            # Dropout(0.5),

            # Dense(units = 84, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            # Dropout(0.5),
            # Dense(units = 2, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            # Activation('softmax')
        ])
    optimizer = Adam(learning_rate = lr, epsilon = 1e-8)
    # optimizer = keras.optimizers.AdamW(learning_rate = lr, )
    # optimizer = keras.optimizers.SGD(learning_rate = lr)
    lr_metric = get_lr_metric(optimizer)
    LeNet.compile(optimizer = optimizer, loss = SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy', lr_metric])
    return LeNet

def create_model_LeNet_5_1D(C = 0.0001, lr = 0.0001, std = 0.01):
    numEpochDataPoints = 128*30
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        initializer = HeNormal()  # Kaiming initializer
        LeNet = Sequential([
            GaussianNoise(stddev = std, input_shape = (numEpochDataPoints, 1)),
            Conv1D(filters = 6, kernel_size = 5, input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),
            AveragePooling1D(pool_size = 2, strides = 2),

            Conv1D(filters = 16, kernel_size = 5, kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),
            AveragePooling1D(pool_size = 2, strides = 2),

            Conv1D(filters = 120, kernel_size = 5, kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),

            Flatten(),
            Dense(units = 84, kernel_regularizer = regularizers.l2(C)),
            Activation('tanh'),
            Dense(units = 2, kernel_regularizer = regularizers.l2(C)),
            Activation('softmax')


        ])
    optimizer = Adam(learning_rate = lr, epsilon = 1e-8)
    # optimizer = keras.optimizers.AdamW(learning_rate = lr, )
    # optimizer = keras.optimizers.SGD(learning_rate = lr)
    lr_metric = get_lr_metric(optimizer)
    LeNet.compile(optimizer = optimizer, loss = SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy', lr_metric])
    return LeNet

def create_test_model():


    testmodel = Sequential([
        preprocessing.Normalization(mean = 0, variance = 1),  # Standardization to 0 mean, 1 std (or 1 variance)
        RandomFlip1D(),
        # RandomGaussianNoise1D(),
        # Rescaling(1./250e-6)
    ])
    return testmodel
    


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
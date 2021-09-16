import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# 5 CNN + 3 Dense layers
def create_model_AlexNet(C = 0.0001, lr = 0.0001):  # Default hyperparameters
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        initializer = tf.keras.initializers.HeNormal()  # Kaiming initializer
        numEpochDataPoints = 128*30
        AlexNet = keras.Sequential([
            # CNN 1
            #keras.layers.Dropout(1),
            keras.layers.Conv1D(filters = 64, kernel_size = 11, strides = 4,  padding = 'same', input_shape = (numEpochDataPoints, 1), kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #keras.layers.Dropout(0.5),
            keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),

            # CNN 2
            keras.layers.Conv1D(filters = 192, kernel_size = 5, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C),data_format = 'channels_last'),
            keras.layers.Activation('relu'),
            #keras.layers.Dropout(0.5),
            keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),

            # CNN 3
            keras.layers.Conv1D(filters = 384, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C),data_format = 'channels_last'),
            keras.layers.Activation('relu'),
            #keras.layers.Dropout(0.5),

            # CNN 4
            keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            keras.layers.Activation('relu'),
            #keras.layers.Dropout(0.5),

            # CNN 5
            keras.layers.Conv1D(filters = 256, kernel_size = 3, strides = 1,  padding = 'same', kernel_initializer=initializer, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C), data_format = 'channels_last'),
            keras.layers.Activation('relu'),
            #keras.layers.Dropout(0.5),
            keras.layers.MaxPooling1D(pool_size = 3, strides = 2, padding = 'valid', data_format = 'channels_last'),
            keras.layers.AveragePooling1D(pool_size = 18, strides = 18, padding = 'valid', data_format = 'channels_last'),
            keras.layers.Flatten(),

            # Dense 1
            keras.layers.Dense(units = 512, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.5),

            # Dense 2
            keras.layers.Dense(units = 128,  kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.5),

            # Dense 3
            keras.layers.Dense(units = 2, kernel_regularizer = regularizers.l2(C), bias_regularizer = regularizers.l2(C)),
            keras.layers.Activation('softmax') # Output activation is softmax
        ])
        
    optimizer = keras.optimizers.Adam(learning_rate = lr, epsilon = 1e-8)
    # optimizer = keras.optimizers.AdamW(learning_rate = lr, )
    # optimizer = keras.optimizers.SGD(learning_rate = 0.001)
    AlexNet.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(),  metrics = ['sparse_categorical_accuracy'])
    #print(AlexNet.summary())
    return AlexNet

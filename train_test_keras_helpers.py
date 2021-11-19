import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare1D(ds, batch_size=128, std = 0.01, shuffle=False, augment=False):
    print("prepping 1D data")
    AUTOTUNE = tf.data.AUTOTUNE

    # Resize and rescale all datasets.
    rescale = tf.keras.Sequential([
        Rescaling(1./250e-6)
    ])
    ds = ds.map(lambda x, y: (rescale(x), y), 
                num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)

    # Use data augmentation only on the training set.
    data_augmentation = tf.keras.Sequential([
        RandomFlip1D(),
        RandomGaussianNoise1D(mean = 0, std = std),
    ])
    
    if augment:
        print("now augmenting")
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    # Reshape to tensor
    reshape = tf.keras.Sequential([
        ReshapeToTensorLayer()
    ])
    ds = ds.map(lambda x, y: (reshape(x), y), 
            num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=AUTOTUNE)

# Add Gaussian noise to 1D time series data with std and mean
class RandomGaussianNoise1D(tf.keras.layers.Layer):
    def __init__(self, p=0.5, mean = 0.0, std=0.01, shape=[3840, 1], **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.mean = mean
        self.std = std  
        self.shape = shape
    
    def call(self, x, training):
        if not training:
            return x
        else:
            return self.noise1D(x)

    def noise1D(self, x):
        if tf.random.uniform([]) < self.p:
            # The noise itself has zero mean, and the signal has mean of 0.5 (due to minmaxscaler [0, 1])
            x += tf.random.normal(self.shape, self.mean, self.std) 
        return x

# Reverse time series data horizontally
class RandomFlip1D(tf.keras.layers.Layer):
    def __init__(self, p=0.5, **kwargs):
        print("initailised")
        super().__init__(**kwargs)
        self.p = p
    def call(self, x, training):
        if not training:
            return x
        else:
            return self.flip1D(x)
        
    def flip1D(self, x):
        if tf.random.uniform([]) < self.p:
            x = x[::-1]
        return x

# Flip image horizontally
class RandomFlip2D(tf.keras.layers.Layer):
    def __init__(self, p = 0.5, **kwargs):
        print("initailised")
        super().__init__(**kwargs)
        self.p = p

    def call(self, x, training):
        if not training:
            return x
        else:
            return self.flip2D(x)
        
    def flip2D(self, x):
        if tf.random.uniform([]) < self.p:
            print("reversing")
            x = x[...,::-1]
        else:
            print("no reverse")
        return x

class RandomCrop2D(tf.keras.layers.Layer):
    def __init__(self, target_width = 227, start_range = 29, **kwargs):
        super().__init__(**kwargs)
        self.target_width = target_width
        self.start_range = start_range

    def call(self, x, training):
        if not training:
            return x
        else:
            return self.crop2D(x)

    def crop2D(self, x):
        start = tf.random.uniform(shape=(), minval=0, maxval=self.start_range, dtype=tf.int32)
        x = x[:, start:start+self.target_width] # Height is not cropped off
        return x


class ReshapeToTensorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        print("initailised")
        super().__init__(**kwargs)

    def call(self, x):
        return reshape(x)

def reshape(x):
    return tf.expand_dims(x, -1)

class CustomDataGenerator(ImageDataGenerator):
    def __init__(self, p = 0.5, **kwargs):
        super().__init__(preprocessing_function = self.flip1D, **kwargs)
        self.p = p

    def flip1D(self, x):
        print(x)
        a = tf.random.uniform([])
        print(a)
        if a < 0.5:
            print("reversing")
            x = x[::-1]
        else:
            print("no reverse done")
        return x
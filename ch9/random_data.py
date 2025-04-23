import jax
import jax.numpy as jnp

'''
so jax handles random numbers a bit differently.
This is driven by its functional nature and by some of the nuance of XLA

numpy keeps it as an internal state, but jax is pure functional, so no state
'''
'''
an aside on PRNG from the book:
    - a prng has period, which is the smallest number of steps after which the prng returns to 
    a previously used value
    mersenne twister is commonly used.
'''
'''
there are some hardware based random number generators. there is also a quantum one that you ca
call an API to via python quantumrandom
'''

import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20,10]

data_dir = '/tmp/tfds'

data, info = tfds.load(name="cats_vs_dogs",
                       data_dir=data_dir,
                       split=["train[:80%]", "train[80%:]"],
                       as_supervised=True,
                       with_info=True)

(cats_dogs_data_train, cats_dogs_data_test) = data

CLASS_NAMES = ['cat', 'dog']
ROWS = 2
COLS = 5

i = 0
fig, ax = plt.subplots(ROWS, COLS)
for image, label in cats_dogs_data_train.take(ROWS*COLS):
    ax[int(i/COLS), i%COLS].axis('off')
    ax[int(i/COLS), i%COLS].set_title(CLASS_NAMES[label])
    ax[int(i/COLS), i%COLS].imshow(image)
    i += 1

plt.show()


HEIGHT = 200
WIDTH = 200
NUM_LABELS = info.features['label'].num_classes

import jax.numpy as jnp

def preprocess(img, label):
    """
    resize and preprocess images
    """
    return tf.image.resize(img, [HEIGHT, WIDTH]) / 255.0, label

train_data = tfds.as_numpy(
        cats_dogs_data_train.map(preprocess).batch(32).prefetch(1))
test_data = tfds.as_numpy(
        cats_dogs_data_test.map(preprocess).batch(32).prefetch(1))



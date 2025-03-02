from os import WEXITSTATUS
from PIL.Image import init
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


plt.rcParams['figure.figsize'] = [10, 5]

data_dir = './data/tfds'

def data_load() -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.dataset_info.DatasetInfo]:
    data, info = tfds.load(name="mnist",
                           data_dir=data_dir,
                           as_supervised=True,
                           with_info=True)
    data_train = data['train']
    data_test = data['test']
    return data_train, data_test, info


def show_data_sample(data: tf.data.Dataset) -> None:
    ROWS = 3
    COLS = 10
    i = 0 
    fig, ax = plt.subplots(ROWS, COLS)
    for image, label in data.take(ROWS*COLS):
        ax[int(i/COLS), i%COLS].axis('off')
        ax[int(i/COLS), i%COLS].set_title(str(label.numpy()))
        ax[int(i/COLS), i%COLS].imshow(np.reshape(image, (28,28)), cmap='gray')
        i += 1

    plt.show()

def preprocess(img: np.array, label: int) -> Tuple[np.array, int]:
    return (tf.cast(img, tf.float32)/255.0), label


def init_network_params(sizes, key=random.PRNGKey(40), scale=1e-2):

    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(b_key, (n,))
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    
if __name__ == "__main__":
    HEIGHT, WIDTH, CHANNELS = 28,28,1
    NUM_PIXELS = HEIGHT * WIDTH * CHANNELS

    data_train, data_test, info = data_load()
    NUM_LABELS = info.features['label'].num_classes
    # show_data_sample(data_train)
    train_data = tfds.as_numpy(data_train.map(preprocess).batch(32).prefetch(1))
    test_data = tfds.as_numpy(data_test.map(preprocess).batch(32).prefetch(1))

    LAYER_SIZES = [28*28, 512, 10]
    PARAM_SCALE = 0.01
    params = init_network_params(LAYER_SIZES, random.PRNGKey(40), scale=PARAM_SCALE)




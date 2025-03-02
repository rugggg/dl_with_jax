import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


plt.rcParams['figure.figsize'] = [10, 5]

data_dir = './data/tfds'

def data_load() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    data, info = tfds.load(name="mnist",
                           data_dir=data_dir,
                           as_supervised=True,
                           with_info=True)
    data_train = data['train']
    data_test = data['test']
    return data_train, data_test


def show_data_sample(data: tf.data.Dataset) -> None:
    ROWS = 3
    COLS = 10
    print(type(data))
    i = 0 
    fig, ax = plt.subplots(ROWS, COLS)
    for image, label in data.take(ROWS*COLS):
        ax[int(i/COLS), i%COLS].axis('off')
        ax[int(i/COLS), i%COLS].set_title(str(label.numpy()))
        ax[int(i/COLS), i%COLS].imshow(np.reshape(image, (28,28)), cmap='gray')
        i += 1

    plt.show()

if __name__ == "__main__":
    data_train, data_test = data_load()
    show_data_sample(data_train)

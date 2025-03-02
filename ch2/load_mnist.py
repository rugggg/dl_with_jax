import tensorflow as tf
import tensorflow_datasets as tfds


data_dir = './data/tfds'

data, info = tfds.load(name="mnist",
                       data_dir=data_dir,
                       as_supervised=True,
                       with_info=True)
data_train = data['train']
data_test = data['test']

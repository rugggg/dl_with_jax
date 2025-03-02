import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random, vmap, grad, value_and_grad, jit
import jax.numpy as jnp
from jax.nn import logsumexp, swish, one_hot
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time


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
        return scale * random.normal(w_key, (n,m)), scale * random.normal(b_key, (n,))

    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    

def predict(params, image):
    activations = image
    for w,b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = swish(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits

batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, images, targets):
    logits = batched_predict(params, images)
    log_preds = logits - logsumexp(logits)
    return -jnp.mean(targets*log_preds)

@jit
def update(params, x, y, epoch_number):
    INIT_LR = 1.0
    DECAY_RATE = 0.95
    DECAY_STEPS = 5
    loss_value, grads = value_and_grad(loss)(params, x, y)
    
    lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
    return [(w - lr * dw, b - lr *db) for (w,b), (dw,db) in zip(params, grads)], loss_value

@jit
def batch_accuracy(params, images, targets):
    images = jnp.reshape(images, (len(images), 28*28*1))
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == targets)

def accuracy(params, data):
    accs = []
    for images, targets in data:
        accs.append(batch_accuracy(params, images, targets))
    return jnp.mean(jnp.array(accs))



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

    random_img = random.normal(random.PRNGKey(42), (28*28*1,))
    preds = predict(params, random_img)

    random_flattened_images = random.normal(random.PRNGKey(41), (32, 28*28*1))

    batched_preds = batched_predict(params, random_flattened_images)

    for epoch in range(10):
        start_time = time.time()
        losses = []
        for x, y in train_data:
            x = jnp.reshape(x, (len(x), NUM_PIXELS))
            y = one_hot(y, NUM_LABELS)
            params, loss_value = update(params, x, y, epoch)
            losses.append(loss_value)
        epoch_time = time.time() - start_time
        start_time = time.time()
        train_acc = accuracy(params, train_data)
        test_acc = accuracy(params, test_data)

        eval_time = time.time() - start_time
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Eval in {:0.2f} sec".format(eval_time))
        print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
        print("Training set accuracy {}".format(train_acc))
        print("test set accuracy {}".format(test_acc))


    



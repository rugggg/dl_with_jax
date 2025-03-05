from jax import random
from jax import lax
import jax.numpy as jnp

def random_augmentation(image, augmentations, rng_key):
    augmentation_index = random.randint(
            key=rng_key, minval=0,
            maxval=len(augmentations), shape=())
    augmented_image = lax.switch(augmentation_index, augmentations, image)
    return augmented_image

augmentations = [# some list of functions goes here
                 ]
image = jnp.zeros((10,10))

new_image = random_augmentation(image, augmentations, random.PRNGKey(40))

# jax.lax does not do type promotion for you

print(jnp.add(42, 42.0))

try:
    lax.add(42, 42.0)
except TypeError as e:
    print(e)

lax.add(jnp.float32(42), 42.0)


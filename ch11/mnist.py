import jax
from jax import random
from flax import linen as nn

class MLP(nn.Module):
    """ simple MLP """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=512)(x)
        x = nn.activation.swish(x)
        x = nn.Dense(features=10)(x)
        return x

model = MLP()

key1, key2 = random.split(random.PRNGKey(0))
random_flattened_image = random.normal(key1, (28*28*1,))
params = model.init(key2, random_flattened_image)
print(jax.tree_util.tree_map(lambda x: x.shape, params))

model.apply(params, random_flattened_image)





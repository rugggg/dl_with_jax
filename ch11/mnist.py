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

def loss(params, images, targets):
    """CCE loss"""
    logits = model.apply(params, images)
    log_preds = logits - jax.nn.logsumexp(logits)
    return -jnp.mean(targets*log_preds)

@jax.jit
def update(params, x, y, epoch_number):
    loss_value, grads = jax.value_and_grad(loss)(params, x, y)
    lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
    return jax.tree_util.tree_map(lambda p, g: p - lt * g, params, grads), loss_value






import jax
import jax.numpy as jnp

def selu(x, alpha=1.6732, scale=1.0507):
    return scale * jnp.where(x>0, x, alpha * jnp.exp(x) - alpha)

# jit can be called via jit(f) or as a decorator @jit

# jit and ahead of time (aot)
# jit does the compilation at the first run of the code to be compiled, so first run
# is the slowest
# AOT on the other hand, compiles everything before running, so its different

x = jax.random.normal(jax.random.PRNGKey(42), (1_000_000,)) # a million random numbers
selu_jit = jax.jit(selu)
selu_jit(x).block_until_ready()

# or
@jax.jit
def selu_j(x, alpha=1.6732, scale=1.0507):
    return scale * jnp.where(x>0, x, alpha * jnp.exp(x) - alpha)

# warmup call 
z = selu_j(x)
selu_j(x).block_until_ready()

# jit gets like a 5x speedup here fwiw

# jit also has a backend param to compile for a specific device, ie cpu/gpu/tpu

selu_jit_cpu = jax.jit(selu, backend="cpu")
selu_jit_gpu = jax.jit(selu, backend="gpu")

# to properly benchmark, you should also place data onto the correct device!
x_cpu = jax.device_put(x, jax.devices('cpu')[0])
# x_gpu = jax.device_put(x, jax.devices('gpu')[0])

# static arguments

def dense_layer(x, w, b, activation_func):
    return activation_func(x*w+b)

x = jnp.array([1.0, 2.0, 3.0])
w = jnp.ones((3,3))
b = jnp.ones(3)

try:
    dense_layer_jit = jax.jit(dense_layer)
    dense_layer_jit(x, w, b, selu) # will fail beacuse of the last arg being a function!
except TypeError as e:
    print(e)


dense_layer_jit = jax.jit(dense_layer, static_argnums=3)
print(dense_layer_jit(x, w, b, selu))

# another note on JIT static parameters is that for each value of a static param, the function being jitted is recompiled!

# if you want to do jit with static args, use functools partial
from functools import partial
@partial(jax.jit, static_argnums=0)
def dist(order, x, y):
    return jnp.power(jnp.sum(jnp.abs(x-y)**order), 1.0/order)





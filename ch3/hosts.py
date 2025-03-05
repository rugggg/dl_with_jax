import jax
from jax._src.api import block_until_ready
import jax.numpy as jnp
import numpy as np

print(jax.devices())
print(jax.devices('cpu'))
print(jax.local_devices())

arr = jnp.array([1, 42, 31337])
print(arr.device)

# you can also use jax.default_device() to set it or JAX_PLATFORMS env or flag
# there is data commited to a device, and data uncommited to a device
# uncommited data goes to default device
# to put onto a specific device:
# NOTE! this does not move data, it is a copy operation!
# jax functions need to be pure
jax.device_put(arr, jax.devices()[0])


# palias is kernel writing library for JAX ala triton

# JAX does async dispatch, so an Array is actually a 'future'
# the value will be produced in the future on the device, but it does 
# have the shape and type of the array already
# if you print or inspect, it forces the conversion to a numpy array
# and so it forces jax to wait and produce the value

# canuse the array.block_until_ready() to force this behavior

# run this in colab: 
'''
a = jnp.array(range(1000000)).reshape((1000, 1000))
print(a.device)
@time x = jnp.dot(a, a)
print(time)
 
@time x = jnp.dot(a, a).block_until_ready()
print(x)
'''

# jax arrays are immutable
a_jnp = jnp.array(range(10))
a_np = np.array(range(10))

print(a_np[5], a_jnp[5])

a_np[5] = 100

try:
    a_jnp[5] = 100
except TypeError as e:
    print(e)
a_jnp.at[5].set(100)
print(a_jnp[5])

# out of bounds handling in jax is slightly different

# notice that this will not fail! it clips to last index
print(a_jnp[42])

# so you should use .at and you can set modes for how to handle out of bounds indexes
print(a_jnp.at[42].get(mode='promise_in_bounds'))
print(a_jnp.at[42].get(mode='drop'))
print(a_jnp.at[42].get(mode='clip'))
print(a_jnp.at[42].get(mode='fill', fill_value=99))


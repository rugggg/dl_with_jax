import jax
import jax.numpy as jnp

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

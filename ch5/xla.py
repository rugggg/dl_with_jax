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

# jit gets like a 5x speedup here fwiw

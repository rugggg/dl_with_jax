import jax
import jax.numpy as jnp
from sympy import lambdify


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



# a huge note to remember!
# because jax jit takes a python func and traces it to jaxpr then xla at first invocation, you have to remember that side effects only happen at first invocation!!!
# additionally, jax NEEDS pure functions, so global state or things not specificlly in the input variables will be set static at the first call, for example, a global variable

global_state = 1

def impure_function(x):
    print(f'side effect: x={x}')
    y = x*global_state
    return y

impure_function_jit = jax.jit(impure_function)

print(impure_function_jit(10)) # side effects happen, and global state is correct

print(impure_function_jit(10)) # will not have print side effect, but global state ok

global_state = 2
print(impure_function_jit(10)) # will not have print side effect, AND global state is not updated!, still outputs 10!


# jaxpr time

def f1(x, y, z):
    return jnp.sum(x + y * z)

x = jnp.array([1.0, 1.0, 1.0])
y = jnp.ones((3,3))*2.0
z = jnp.array([2.0, 1.0, 0.0]).T

print(jax.make_jaxpr(f1)(x,y,z))

# ok skipping most jaxpr stuff for now, might come back to it
# what is interesting is that jax traces functions to compile them w/jit
# jaxpr on first trace triggers side effect, but never traces it. hence the
# behavior above where global variables dont get changed

# control structures under JIT a bit more interesting
# if they do not depend on a global variable and do not depend on any 
# input value, no worries. They can depend on shape of input however

def f3(x):
    y = x
    for i in range(5):
        y += i
    return y

print(jax.make_jaxpr(f3)(0)) # no input dependence at all, no worries

def f4(x):
    y = 0
    for i in range(x.shape[0]):
        y += x[i]
    return y
print(jax.make_jaxpr(f4)(jnp.array([1., 2., 3.]))) # gets unrolled, depends on shape of input but thats ok

def relu(x):
    if x > 0:
        return x
    return 0.0

print(relu(10.)) # ok

# print(jax.make_jaxpr(relu)(10.0)) # jit/jaxpr will fail on this bc relu func depends on input value!
# however, we can use static_argnums to declare it static.
# this works, but costs a recompile each time a new value is called

print(jax.make_jaxpr(relu, static_argnums=0)(10.0)) 

# but relu will see a new value like all the time! this is the use of jax.lax control flow!
def f5(x):
    return jax.lax.fori_loop(0, x, lambda i,v: v+i, 0)

def relu(x):
    return jax.lax.cond(x>0, lambda x: x, lambda x: 0.0, x)

relu(12.3)
print(jax.make_jaxpr(relu)(12.3))

# great! now the code is through to jaxpr, but it still needs to get to XLA
# so a key takeaway: jax.lax for control flow in jit.

# So XLA was created at Google for tensorflow graph JIT compilation and fusing. 
# for example:
def f(x, y, z):
    return jnp.sum(x + y * z)
# has three seperate ops in it. XLA will optimize and fuse this to a single op.
# it also removes extra intermediate varoables and can keep data in same memory/gpu location
# xla compiles to native backend code for accelerator of choice
# we refer to the system sending data to XLA as the frontend - TensorFlow, PyTorch, JAX, Julia etc. So I am a front end developer!

# ok wow a ton on XLA and compilers in this book here, very deep rabbit hole
x = jnp.array([1.0, 1.0, 1.0])                             
y = jnp.ones((3,3))*2.0                                           
z = jnp.array([2.0, 1.0, 0.0]).T


f_jitted = jax.jit(f)
f_lowered = f_jitted.lower(x,y,z)
print(f_lowered.as_text())
f_compiled = f_jitted.lower(x,y,z).compile()
print(f_compiled.as_text())

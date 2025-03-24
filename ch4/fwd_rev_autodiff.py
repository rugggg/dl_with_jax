import jax
import jax.numpy as jnp
# well apparently this is the hardest part of the book! lets see what we have in store

# Evaluation Trace - a computation can be represented as an evaluation trace of elementrary operations, also known as a Wengert list or a tape. 
# wengert list ignores control flow type operation 

# take example of 
# f(x1, x2) = x1 * x2 + sin(x1 * x2)
'''
you would end up with the forward mode taking both a vi value for each intermediate step and also a vi', or vi/dx 
you then get a primal (vi) and a tanget(vi') for each step
this is the dual numbers approach
'''
def f(x1, x2):
    return x1*x2 + jnp.sin(x1*x2)

x = (7.0, 2.0)

print(jax.grad(f)(*x))

# jax uses operator overloading ti implement this fwiw

# in this method, you need to run a forward pass for each input variable, ie run once for
# x1, and again for x2
# this is giving a single column for a jacobian matrix
# so this is why jacfwd works better for functions with less inputs than outputs!

# directional diffs: allow to get derivative in any direction, not just positive direction of axis
# to do so, specify a direction vector as initial value for tangents

# jacobian vector product
# jvp takes in a function to be diffed, primals values at which the jacobian should be evaluated and a vector of tangents for whch jvp should be evaluated
# returning primals and tangets

def f2(x):
    return [
            x[0]**2 + x[1]**2 - x[1]*x[2],
            x[0]**2 - x[1]**2 + 3*x[0]*x[2]
    ]

x = jnp.array([3.0, 4.0, 5.0])
v = jnp.array([1.0, 1.0, 1.0])
# pack the jnp array into a tuple to make jvp happy
p,t = jax.jvp(f2, (x,), (v,))
print(p)
print(t)
# now if you wanted to recover individual columns of jacobian - pass unit vectors!
p,t = jax.jvp(f2, (x,), (jnp.array([1.0, 0.0, 0.0]),))
print(t)
p,t = jax.jvp(f2, (x,), (jnp.array([0.0, 1.0, 0.0]),))
print(t)
p,t = jax.jvp(f2, (x,), (jnp.array([0.0, 0.0, 1.0]),))
print(t)

# so therefore, jvp can help to check manual derivatives as well

# Reverse Mode and VJP
# so reverse mode we have two phases,
# first phase is that function is run fwd and intermediate values are stored.
# all dependencies also are recorded
# Then, in 2nd phase, each intermediate variable is complemented with an adjoint, or cotangent
# vi' = dyi/dvi -> the derivative of the jth output yj with repsect to vi that represente sensitivity of output yj to changes in vi
# these are calculated by going backward and propogating adjoints in reverse, from outputs to inputs

def f(x1, x2):
    return x1*x2 + jnp.sin(x1*x2)
x = (7.0, 2.0)

# take gradients with respect to both params
print(jax.grad(f, argnums=(0,1))(*x))

# so we've gotten gradients for both inputs with a single pass
# and as can be seen, in ml, the case with many inputs and few outputs is much more common, so this is often much more useful

# and of course, there is a vjp corresponding pair to jvp!
# vjp takes a function, an input vector and returns 
# the output of f(x) and a function to evaluate VJP with a given adjoint for the backwards phase

p, vjp_func = jax.vjp(f, *x)

print(p)

print(vjp_func(1.0))

x = jnp.array([3.0, 4.0, 5.0])
p, vjp_func = jax.vjp(f2, x)

print(p)
print(vjp_func([1.0, 0.0]))
print(vjp_func([0.0, 1.0]))

# ok so that mostly tracks for me, a little in the weeds here, but good to go through the process here. Will come back if I need more depth here!

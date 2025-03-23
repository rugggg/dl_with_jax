import jax
import jax.numpy as jnp

# a jacobian matrix is a matrix containing all partial derivatives for a vector valued function of
# several values

# for a deep learning model, this would be the partial derivative of the loss to all params. you can then express this as a product of jacobian matrices from the output of one layer and the weights 

# jax has a jacfwd and a jacrev to calculate in fwd/rev autodiff respectively 

def f(x):
    return [
            x[0]**2 + x[1]**2 - x[1]*x[2], # first output
            x[0]**2 - x[1]**2 + 3*x[0]*x[2] # second output
        ]

print(jax.jacrev(f)(jnp.array([3.0, 4.0, 5.0])))


print(jax.jacfwd(f)(jnp.array([3.0, 4.0, 5.0])))


# hessian matrixes
# to get a second order derivative of a function with multiple inputs, we need to obtain a hessian matrix

def f(x):
    return x[0]**2 - x[1]**2 + 3*x[0]*x[2]

jax.hessian(f)(jnp.array([3.0, 4.0, 5.0]))
# equivalent to
def hessian(x):
    return jax.jacfwd(jacrev(f))



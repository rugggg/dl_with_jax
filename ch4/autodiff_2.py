import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10*np.pi, num=1000)
e = np.random.normal(scale=10.0, size=x.size)
y = 65 + 1.8*x + 40*np.cos(x) + e

plt.scatter(x, y)
plt.show()

import jax
import jax.numpy as jnp

xt = jnp.array(x)
yt = jnp.array(y)

# jax does autodiff via jax.grad()
# this is again, functional returning a differentiated function w/respect to the 
# first parameter

learning_rate = 1e-2
model_parameters = jnp.array([1., 1.])

def model(theta, x):
    w, b = theta
    return w * x + b

def loss_fn(model_parameters, x, y):
    prediction = model(model_parameters, x)
    return jnp.mean((prediction-y)**2)

grads_fn = jax.grad(loss_fn)
grads = grads_fn(model_parameters, xt, yt)
model_parameters -= learning_rate * grads

def dist(order, x, y):
    return jnp.power(jnp.sum(jnp.abs(x-y)**order),1.0/order)

# setting diff w/repsect to parameter other than first
dist_d_x = jax.grad(dist, argnums=1)

print(dist_d_x(1, jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])))

dist_d_xy = jax.grad(dist, argnums=(1,2)) # multiple param diffs!

print(dist_d_x(1, jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])))


# another cool thing about jax.grad is that it returns the grads as the same shape as the input!
# so if we used a dict instead of a tuple, we get a dict back!
model_parameters = {'w': jnp.array([1.]), 'b': jnp.array([1.])}

def model(param_dict, x):
    w, b = param_dict['w'], param_dict['b']
    return w * x + b

def loss_fn(model_parameters, x, y):
    prediction = model(model_parameters, x)
    return jnp.mean((prediction-y)**2)

grads_fn = jax.grad(loss_fn)
grads = grads_fn(model_parameters, xt, yt)
print(grads)


model_parameters = jnp.array([1., 1.])
def model(theta, x):
    w, b = theta
    return w * x + b

def loss_fn(model_parameters, x, y):
    prediction = model(model_parameters, x)
    return jnp.mean((prediction-y)**2), prediction # adds prediction output as well
# we need to update the grad call the account for this via has_aux arg

grads_fn = jax.grad(loss_fn, has_aux=True)
grads, preds = grads_fn(model_parameters, xt, yt)
model_parameters -= learning_rate * grads

# jax also has the ability to do value_and_grad instead of just grad so that you can 
# get outout of the function and the grad

grads_fn = jax.value_and_grad(loss_fn, has_aux=True)
(loss, preds), grads = grads_fn(model_parameters, xt, yt)
model_parameters -= learning_rate * grads

# per sample gradients are also an option

# and you can stop gradients in the graph at certain points if desired

def f(x, y):
    return x**2+jax.lax.stop_gradient(y**2)
jax.grad(f, argnums=(0,1))(1.0, 1.0)


# higher order derivatives
def f(x):
    return x**4 + 12*x + 1/x
f_d1 = jax.grad(f)
f_d2 = jax.grad(f_d1)
f_d3 = jax.grad(f_d2)

x = 11.0

print(f_d1(x))
print(f_d2(x))
print(f_d3(x))

def f(x):
    return x**3 + 12*x +7*x*jnp.sin(x)

x = np.linspace(-10, 10, num=500)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(x, f(x), label = r"$y = x^3 + 12x +7x*sin(x)$")

df = f
for d in range(3):
    df = jax.grad(df)
    ax.plot(x, jax.vmap(df)(x),
            label=f"{['1st', '2nd', '3rd'][d]} derivative")
    ax.legend()

plt.show()

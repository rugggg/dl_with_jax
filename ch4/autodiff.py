# working on getting gradients in jax via autograd

# start with manual dx
def f(x):
    return x**4 + 12*x + 1/x
def df(x):
    return 4*x**3 + 12 -1/x**2

x = 40
print(f(x), df(x))
# easy enuogh a closed form solution there
# but like for a NN, the size of the actual eq and the derivate would be a freaking book
# so use symbolic diffs
import sympy
x_sym = sympy.symbols('x')
f_sym = f(x_sym)
df_sym = sympy.diff(f_sym)
print(f_sym)
print(df_sym)
# actually sympy is pretty cool
f = sympy.lambdify(x_sym, f_sym)
print(f(x))
df = sympy.lambdify(x_sym, df_sym)
print(df(x))
# so even with all that, we are still in trouble for large nets 
# we have non closed form experssions (control flow etc)
# and the experssion would be massive

# so lets use numerical differntiation

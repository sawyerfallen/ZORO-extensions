import jax
import jax.numpy as jnp
import functools

from jax import random
from jax import make_jaxpr


def sing_val_sum(X, r):
    XTX = X.transpose()@X
    singvals = jnp.flip(jnp.real(jnp.linalg.eigvalsh(XTX)))
    return 0.5*jnp.sum(singvals[:r])

def sing_val_sum_grad(X):
    r = 3
    U, S, VH = jnp.linalg.svd(X, full_matrices=True)
    SIGR = S.at[r:].set(0)

    return U@jnp.diag(SIGR)@VH

def make_linear_data(A,n=100,seed=42,noiselevel=1e-4):
    key = random.PRNGKey(seed)
    assert len(A.shape) == 2

    x = random.normal(key, shape=(A.shape[1],n))
    b = A @ x + noiselevel * random.normal(key, shape=(A.shape[0],n))

    jnp.savez('lin',x=x,b=b)

def linear_least_squares_test(A):
    file = jnp.load('lin.npz')

    x = file['x']
    b = file['b']

    return jnp.linalg.norm(A@x - b)**2


def make_DNN_data(A,B,n=100,seed=42,noiselevel=1e-4):
    key = random.PRNGKey(seed)
    assert len(A.shape) == 2

    x = random.normal(key, shape=(A.shape[1],n))
    b = B@jax.nn.relu(A @ x) + noiselevel * random.normal(key, shape=(A.shape[0],n))

    jnp.savez('dnn',x=x,b=b)

def DNN_test(A,B):
    file = jnp.load('dnn.npz')

    x = file['x']
    b = file['b']

    return jnp.linalg.norm(B@jax.nn.relu(A @ x) - b)**2

            

seed = 8566
key = random.PRNGKey(seed)
X = random.normal(key, shape=(5,5))
print(X)

print(sing_val_sum(X, 3))

#gradjax_sing_val_sum = jax.grad(sing_val_sum)
svs = jax.jit(functools.partial(sing_val_sum, r=3))

svs_jit = jax.jit(svs)

#Y = random.normal(key, shape=(10,10))

print(svs_jit(X))


print("TESTING linear_least_squares_test:")
print(80*"-")


A0 = random.normal(key, shape=(10,5))
A1 = random.normal(key, shape=(5,10))


make_linear_data(A0 @ A1)

jit_lin = jax.jit(linear_least_squares_test)
jit_lingrad = jax.jit(jax.grad(jit_lin))

print('test jit objective output:')
print(jit_lin(0 * jnp.eye(10)))
print('test jit grad output:')
print(jit_lingrad(0 * jnp.eye(10)))

print("TESTING DNN_test:")
print(80*"-")

B0 = random.normal(key, shape=(10,5))
B1 = random.normal(key, shape=(5,10))

make_DNN_data(A0 @ A1, B0 @ B1)

jit_dnn = jax.jit(DNN_test)
jit_dnngrad = jax.jit(jax.grad(DNN_test, argnums=(0,1)))

print('test jit objective output:')
print(jit_dnn(jnp.eye(10), jnp.eye(10)))
print('test jit grad output (should be two matrices):')
print(jit_dnngrad(jnp.eye(10), jnp.eye(10)))

#print(gradjax_sing_val_sum(X))
#print(sing_val_sum_grad(X))
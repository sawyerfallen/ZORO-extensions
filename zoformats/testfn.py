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


            

seed = 8566
key = random.PRNGKey(seed)
X = random.normal(key, shape=(5,5))
#print(X)

#print(sing_val_sum(X, 3))

#gradjax_sing_val_sum = jax.grad(sing_val_sum)
svs = jax.jit(functools.partial(sing_val_sum, r=3))

svs_jit = jax.jit(svs)

#Y = random.normal(key, shape=(10,10))

#print(svs_jit(X))

#print(gradjax_sing_val_sum(X))
#print(sing_val_sum_grad(X))
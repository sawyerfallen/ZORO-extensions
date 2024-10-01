import jax
import jax.numpy as jnp
from jax import random

#A is list of matricies of the same size as X
def A(X, A):
    y = jnp.sum(A * X, axis=(1, 2))
    return y

def lossA(X, A, b) :
    pred = A(X, A)
    res = b - pred
    return jnp.linalg.norm(res) ** 2



#alternating projections
#y is vector


def altProj(y, A, r, iters):
    
    #initialization from Jain et al
    
    y_r = y[:r]
    A_r = A[:r]
    absum = jnp.sum(y_r[:, None, None] * A_r, axis=0)
    U, S, VH = jnp.linalg.svd(absum, full_matrices=True)
    V = 

    for i in range(iters):
        V = jnp.linalg.lstsq

    return 
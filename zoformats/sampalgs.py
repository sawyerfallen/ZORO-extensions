import jax
import jax.numpy as jnp
from jax import random

#A is num_samples * m*n  where X is m*n
def A(X, A):
    y = jnp.sum(A * X, axis=(1, 2))
    return y

def argMinVT(A, U, y):
    #A shape is num_samples * m * n
    #U is m*r
    #IkU is nm * nr
    #Avecs is num_samples * mn
    #VTvec is rn
    #VT is r * n

    #AIkU is size samp * nr


    IkU = jnp.kron(jnp.identity(A.shape[2]), U)
    Avecs = A.reshape(A.shape[0], A.shape[1]*A.shape[2])
    AIkU = jnp.matmul(Avecs, IkU)
    VTvec = jnp.linalg.lstsq(AIkU, y)
    return VTvec.reshape(U.shape[1], A.shape[2])

    
def argMinU(A, VT, y):
    VkI = jnp.kron(VT.transpose(), jnp.identity(A.shape[1]))
    Avecs = A.reshape(A.shape[0], A.shape[1]*A.shape[2])
    AVkI = jnp.matmul(Avecs, VkI)
    Uvec = jnp.linalg.lstsq(AVkI, y)
    return Uvec.reshape(A.shape[1], VT.shape[0])

#alternating projections
#y is vector


def altProj(y, A, r, iters):
    
    #initialization from Jain et al
    
    y_r = y[:r]
    A_r = A[:r]
    absum = jnp.sum(y_r[:, None, None] * A_r, axis=0)
    U, S, VH = jnp.linalg.svd(absum, full_matrices=True)

    for i in range(iters):
        VT = argMinVT(A, U, y)
        U = argMinU(A, VT, y)
    

    return U@VT
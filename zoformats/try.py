import jax
import jax.numpy as jnp
from jax import random


seed = 8566
key = random.PRNGKey(seed)
U = random.randint(key, shape=(4,3), minval=1, maxval=5)
V = random.randint(key, shape=(3, 4), minval=1, maxval=5)

X = U@V
Z = random.randint(key, shape=(5, 4, 4), minval=1, maxval=5)

def transpose_multiply_trace(matrix):
    # Transpose the matrix
    transposed_matrix = jnp.transpose(matrix)
    # Multiply the transposed matrix with B
    result = jnp.matmul(transposed_matrix, X)
    # Compute the trace of the resulting matrix
    return jnp.trace(result)

# Vectorize the function across the first axis (5 matrices)
batched_operation = jax.vmap(transpose_multiply_trace)
y = batched_operation(Z)

Avecs = Z.reshape(Z.shape[0], Z.shape[1]*Z.shape[2])
UTkI = jnp.kron(U.transpose(), jnp.identity(Z.shape[2]))


VkI = jnp.kron(jnp.identity(Z.shape[1]), V.transpose())
print(y)

print(Avecs@(X.ravel()))

print((Avecs@VkI)@U.ravel())


print(Avecs@UTkI.transpose()@V.ravel())

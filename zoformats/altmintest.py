import jax
import jax.numpy as jnp
from jax import random
import sampalgs

seed = 8566
key = random.PRNGKey(seed)
U = random.normal(key, shape=(10,2))
V = random.normal(key, shape=(2, 10))

X = U@V
key, subkey = random.split(key)

Z = random.normal(subkey, shape=(5, 10, 10))

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
print(y.shape)

est = sampalgs.altProj(y, Z, 2, 100)

print(X-est)
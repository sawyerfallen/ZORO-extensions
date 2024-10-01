import jax
import jax.numpy as jnp
import testfn
import functools
from jax import random

#params

#step size
alpha = 0.5

#iterations
K = 1000

#dimension
m = 5

#rank
r = 3

#number of sampling directions
num_samples = 3

h = 0.0001

#function
f = jax.jit(functools.partial(testfn.sing_val_sum, r = r))


#randomly initialize x0
seed = 8566
key = random.PRNGKey(seed)
X = random.normal(key, shape=(m,m))

#norm vector
normvec = [0] * K
normvec[0] = jnp.linalg.norm(X)

for k in range(1, K+1):
    key, subkey = random.split(key)

    Z = random.normal(subkey, shape=(num_samples, m, m))
    
    X_plus_hZ = X[None, :, :] + h * Z
    
    
    f_vmap = jax.vmap(f, in_axes=0)

    f_XhZ = f_vmap(X_plus_hZ)  
    f_X = f(X) 
    
    y = (f_XhZ - f_X) / h 

    



#low rank alg - alternating projections

#low rank alg - iht


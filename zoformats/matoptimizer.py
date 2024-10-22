import jax
import jax.numpy as jnp
import matplotlib.pyplot
import testfn
import functools
import sampalgs
import math
from jax import random
import matplotlib

#params

#step size
alpha = 0.05

#iterations
K = 1000

#dimension
m = 1000
n = 1000

#rank
r = 2

#number of sampling directions
num_samples = 6

h = 0.001

tol = 10**(-6)

#sampling algorithm iterations
altProjiters = 100

#function
f = jax.jit(functools.partial(testfn.sing_val_sum, r = r))


#randomly initialize x0
seed = 8566
key = random.PRNGKey(seed)
X = random.normal(key, shape=(m,n))

#norm vector
fnvalvec = [0] * K
fnvalvec[0] = f(X)

for k in range(1, K+1):
    print(k)
    key, subkey = random.split(key)

    Z = random.normal(subkey, shape=(num_samples, m, n))
    
    X_plus_hZ = X[None, :, :] + h * Z
    
    
    
    f_vmap = jax.vmap(f, in_axes=0)

    f_XhZ = f_vmap(X_plus_hZ)  
    f_X = f(X) 

    
    y = (f_XhZ - f_X) / h 
    #print(y.shape)


    #low rank alg - alternating projections
    gradfEst = sampalgs.altProj(y, Z, r, altProjiters)
    

    #low rank alg - iht
    X = X - alpha*gradfEst
    fnvalvec[k] = f(X)
    print(f(X))
    if(f(X) < tol):
        break

#matplotlib.pyplot(gradvec)

output = X

print(output)
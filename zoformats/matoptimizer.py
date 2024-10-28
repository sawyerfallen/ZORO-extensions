import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
m = 10
n = 10

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
fnvalvec = [0] * (K+1)
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
plt.plot(fnvalvec)
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("ZO for matricies")
hyperparams_text = f"Iterations: {k}\nMatrix Size: ({n}, {m})\nSampled Directions: {num_samples}\nh: {h}\nRank: {r}\nAlternating projection Iterations: {altProjiters}"
plt.text(len(fnvalvec) + 0.5, max(fnvalvec), hyperparams_text, ha='right', va='top', fontsize=10, bbox=dict(facecolor='lightgrey', alpha=0.5))


plt.show()
output = X

print(output)
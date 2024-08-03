'''
This module contains various test functions for the ZORO algorithm. 
All of them exhibit gradient sparsity or compressibility.

'''
import numpy as np
#import sys
import math
from tensor_utils import *

class SparseQuadric(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.s = s
        self.dim = n
        self.rng = np.random.RandomState()
        
    def __call__(self,x):
        f_no_noise = np.dot(x[0:self.s],x[0:self.s])
        return f_no_noise + self.noiseamp*self.rng.randn()

    
class MaxK(object):
    '''An implementation of the max-k-squared-sum function.'''
    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.dim = n
        self.s = s
        self.rng = np.random.RandomState()

    def __call__(self, x):

        idx = np.argsort(np.abs(x))
        idx2 = idx[self.dim-self.s:self.dim]
        f_no_noise = np.dot(x[idx2], x[idx2])/2
        return f_no_noise + self.noiseamp*self.rng.randn()
 

class CompressibleQuadric(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, n, decay_factor, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.decay_factor = decay_factor
        self.dim = n
        self.rng = np.random.RandomState()
        self.diag = np.zeros(n)
        for i in range(0,n):
            self.diag[i] = math.exp(-self.decay_factor*i)
        
    def __call__(self,x):
        #f_no_noise = 0
        #for i in range(0,self.dim):
            #f_no_noise += math.exp(-self.decay_factor*i)*x[i]**2
        f_no_noise = np.dot(self.diag * x, x)
        return f_no_noise + self.noiseamp*self.rng.randn()
        #f_no_noise = np.dot(x[0:self.s],x[0:self.s])
        #f_no_noise += 1e-4*np.dot(x[self.s:self.dim],x[self.s:self.dim])
        #return f_no_noise + self.noiseamp*self.rng.randn()
    
class SingularSS(object):
    '''An implementation of the sum of squares of r singular values.'''
    def __init__(self, in_shape, r, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(np.prod(in_shape))
        self.in_shape = in_shape
        self.r = r
        self.rng = np.random.RandomState()

    def __call__(self, x):
        assert x.shape == self.in_shape
        singular_vals = np.linalg.svd(x, compute_uv=False)
        f_no_noise = np.dot(singular_vals[:self.r], singular_vals[:self.r])/2
        return f_no_noise + self.noiseamp*self.rng.randn()
    
    def grad(self, x):
        assert x.shape == self.in_shape
        U, s, Vh = np.linalg.svd(x)
        return U[:, :self.r] @ np.diag(s[:self.r]) @ Vh[:self.r, :]
    
    def fwd(self,x,v):
        assert x.shape == self.in_shape
        assert v.shape == self.in_shape
        return self(x), contract_first_indices(v, self.grad(x))
    

if __name__ == '__main__':
    # Parameters
    n = 100
    s = 10
    decay_factor = 0.1
    noiseamp = 1.0
    in_shape = (10, 10)
    r = 5

    # Random input
    x1 = np.random.randn(n)
    x2 = np.random.randn(*in_shape)
    v = np.random.randn(*in_shape)

    # Initialize functions
    sq = SparseQuadric(n, s, noiseamp)
    mk = MaxK(n, s, noiseamp)
    cq = CompressibleQuadric(n, decay_factor, noiseamp)
    ss = SingularSS(in_shape, r, noiseamp)

    # Test cases
    print("SparseQuadric Test Output: ", sq(x1))
    print("MaxK Test Output: ", mk(x1))
    print("CompressibleQuadric Test Output: ", cq(x1))
    print("SingularSS Test Output: ", ss(x2))
    print("SingularSS grad Test Output: ", ss.grad(x2))
    print("SingularSS fwd Test Output: ", ss.fwd(x2,v))
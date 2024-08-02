import numpy as np
import matplotlib.pyplot as plt

from benchmarkfunctions import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

def contract_first_indices(x,T):
   xstr = ''.join([chr(65+i) for i in range(len(x.shape))])
   Tstr = ''.join([chr(65+i) for i in range(len(T.shape))])

   return np.einsum(xstr+','+Tstr+'->'+Tstr[len(xstr):], x, T)

def contract_last_indices(x,T):
   Tstr = ''.join([chr(65+i) for i in range(len(T.shape))])
   xstr = Tstr[-len(x.shape):]

   return np.einsum(xstr+','+Tstr+'->'+Tstr[:len(xstr)], x, T)
   

def plot_derivative_check(f,grad,x,v,title=None, print_log = False, show = False, fname = None): 
    max_iters = 32
    h = np.zeros(max_iters)
    err0 = np.zeros(max_iters)
    err1 = np.zeros(max_iters)

    for i in range(max_iters):
      h[i] = 2**(-i) # halve our stepsize every time

      fv = f(x + h[i]*v)
      T0 = f(x)

      T1 = T0 + h[i] * contract_first_indices(v, grad(x))

      err0[i] = np.linalg.norm(fv - T0) # this error should be linear
      err1[i] = np.linalg.norm(fv - T1) # this error should be quadratic

      if print_log:
          print('h: %.3e, \t err0: %.3e, \t err1: %.3e' % (h[i], err0[i], err1[i]))
        
        
    if title != None:
        plt.title(title)
    
    plt.loglog(h, err0, linewidth=3)
    plt.loglog(h, err1, linewidth=3)
    plt.legend(['$\|f(x) - T_0(x)\|$', '$\|f(x)-T_1(x)\|$'], fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    if show:
        plt.show()

if __name__ == '__main__':
    # Parameters
    n = 100
    s = 10
    noiseamp = 0.0
    in_shape = (10, 10)
    r = 5

    # Random input
    x = np.random.randn(*in_shape)
    v = np.random.randn(*in_shape)

    # Initialize functions
    ss = SingularSS(in_shape, r, noiseamp)

    # Test cases
    plot_derivative_check(ss,ss.grad,x,v,title="Check Derivative for Sum of Squared Singular Values", print_log = True, show = False, fname = "results/SingularSum.png")
import numpy as np

def contract_first_indices(x,T):
   xstr = ''.join([chr(65+i) for i in range(len(x.shape))])
   Tstr = ''.join([chr(65+i) for i in range(len(T.shape))])

   return np.einsum(xstr+','+Tstr+'->'+Tstr[len(xstr):], x, T)

def contract_last_indices(x,T):
   Tstr = ''.join([chr(65+i) for i in range(len(T.shape))])
   xstr = Tstr[-len(x.shape):]

   return np.einsum(xstr+','+Tstr+'->'+Tstr[:len(xstr)], x, T)
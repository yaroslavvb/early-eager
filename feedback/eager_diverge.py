# requirements: MKL, scipy, gzip

import util as u
from util import t
u.check_mkl()

import numpy as np
import scipy
import tensorflow as tf
import time

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

dtype = np.float32

def nonlin(x): return x
def d_nonlin(x): return x

def loss_and_grad(Wf):
  """Returns cost, gradient for current parameter vector."""
  global fs, X, global_cov_A, global_whitened_A
  
  W = u.unflatten(Wf, fs[1:])   # perftodo: this creates transposes
  W.insert(0, X)

  A = [None]*(n+2)
  A[1] = W[0]
  #  print('A[1]', A[1])
  for i in range(1, n+1):
    A[i+1] = nonlin(W[i] @ A[i])
    #    print('i+1', A[i+1])
  err = (A[n+1] - A[1])
  #  print('err', err)

  B = [None]*(n+1)
  B[n] = err*d_nonlin(A[n+1])
#  print('B[n]', B[n])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    B[i] = backprop*d_nonlin(A[i+1])
    #    print('B[i]', B[i])

  dW = [None]*(n+1)
    
  for i in range(1,n+1):
    dW[i] = (B[i] @ t(A[i]))
    print('grad in', B[i])
    print('grad out', (B[i] @ t(A[i])))

  loss = u.L2(err)

  print('dW', dW)
  grad = u.flatten(dW[1:])
  return loss, grad

def main():
  global fs, X, n, f, dsize, lambda_
  
  np.random.seed(1)
  tf.set_random_seed(1)
  
  lr = 0.2
  dsize = 2
  fs = [dsize, 2, 2]  # layer sizes
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2
  train_images = np.asarray([[0, 1], [2, 3]]).astype(dtype)
  X = tf.constant(train_images[:,:dsize].astype(dtype))


  # transpose because flatten does column-wise vectorization
  W0_0 = np.asarray([[0., 1], [2, 3]]).astype(dtype)/10
  #  W1_0 = np.asarray([[4., 5], [6, 7]]).astype(dtype).T/10
  #  W0f = u.flatten([W0_0, W1_0])
  W0f = u.flatten([W0_0])
  Wf = tf.constant(W0f)
  assert Wf.dtype == tf.float32

  losses = []
  for step in range(2):
    loss, grad = loss_and_grad(Wf)
    loss0 = loss.numpy()
    print("Step %3d loss %10.9f"%(step, loss0))
    losses.append(loss0)

    Wf-=lr*grad

if __name__=='__main__':
  main()

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


# for line profiling
try:
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.

dtype = np.float32
tf_dtype = tf.float32


identity_cache = {}
def Identity(n):
  """Identity matrix of size n."""
  global identity_cache
  if n in identity_cache:
    return identity_cache[n]
  else:
    with tf.device('/cpu:0'):
      val = tf.diag(tf.ones((n,), dtype=dtype))
    val = tf.identity(val)   # move to current device
    identity_cache[n] = val
  return identity_cache[n]

def regularized_inverse(mat, lambda_):
  n = int(mat.shape[0])
  assert n == int(mat.shape[1])
  regmat = mat + lambda_*Identity(n)
  return tf.constant(scipy.linalg.inv(regmat.numpy()))


global_cov_A = None
global_whitened_A = None
@profile
def loss_and_grad(Wf):
  """Returns cost, gradient for current parameter vector."""
  global fs, X, global_cov_A, global_whitened_A
  
  W = u.unflatten(Wf, fs[1:])   # perftodo: this creates transposes
  W.insert(0, X)

  A = [None]*(n+2)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = tf.sigmoid(W[i] @ A[i])
  err = (A[3] - A[1])

  def d_sigmoid(y):
    return y*(1-y)

  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  sampled_labels = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  B2[n] = sampled_labels*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    backprop2 = t(W[i+1]) @ B2[i+1]
    B[i] = backprop*d_sigmoid(A[i+1])
    B2[i] = backprop2*d_sigmoid(A[i+1])

  dW = [None]*(n+1)
  pre_dW = [None]*(n+1)  # preconditioned dW

  cov_A = [None]*(n+1)    # covariance of activations[i]
  whitened_A = [None]*(n+1)    # covariance of activations[i]
  cov_B2 = [None]*(n+1)   # covariance of synthetic backprops[i]
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)

  if global_cov_A is None:
    global_cov_A = A[1]@t(A[1])/dsize
    global_whitened_A = regularized_inverse(global_cov_A, lambda_) @ A[1]
    
  cov_A[1] = global_cov_A
  whitened_A[1] = global_whitened_A
    
  for i in range(1,n+1):
    if i > 1:
      cov_A[i] = A[i]@t(A[i])/dsize
      whitened_A[i] = regularized_inverse(cov_A[i], lambda_) @ A[i]
    cov_B2[i] = B2[i]@t(B2[i])/dsize
    whitened_B = regularized_inverse(cov_B2[i], lambda_) @ B[i]
    pre_dW[i] = (whitened_B @ t(whitened_A[i]))/dsize
    dW[i] = (B[i] @ t(A[i]))/dsize

  reconstruction = u.L2(err) / (2 * dsize)
  loss = reconstruction

  grad = u.flatten(dW[1:])
  kfac_grad = u.flatten(pre_dW[1:])
  return loss, grad, kfac_grad

def main():
  global fs, X, n, f, dsize, lambda_
  
  np.random.seed(0)
  tf.set_random_seed(0)
  
  with tf.device('/gpu:0'):
    train_images = u.get_mnist_images()
    dsize = 10000
    fs = [dsize, 28*28, 196, 28*28]  # layer sizes
    lambda_=3e-3
    def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
    n = len(fs) - 2
    X = tf.constant(train_images[:,:dsize].astype(dtype))


    W0_0 = u.ng_init(fs[2],fs[3])
    W1_0 = u.ng_init(fs[3], fs[2])
    W0f = u.flatten([W0_0.flatten(), W1_0.flatten()])
    Wf = tf.constant(W0f)
    assert Wf.dtype == tf.float32
    lr = tf.constant(0.2)

    losses = []
    for step in range(40):
      loss, grad, kfac_grad = loss_and_grad(Wf)
      loss0 = loss.numpy()
      print("Step %d loss %.2f"%(step, loss0))
      losses.append(loss0)
      
      Wf-=lr*kfac_grad
      if step >= 4:
        assert loss < 17.6
      u.record_time()


  u.summarize_time()
  assert losses[-1]<0.59
  assert losses[-1]>0.57
  assert 20e-3<min(u.global_time_list)<50e-3, "Time should be 30ms on 1080"

if __name__=='__main__':
  main()

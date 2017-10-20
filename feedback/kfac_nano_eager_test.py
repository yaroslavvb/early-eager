import util as u
from util import t
u.check_mkl()

SYNTHETIC_LABELS = True

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

import common_gd
args = common_gd.args
args.cuda = not args.no_cuda and (tfe.num_gpus() > 0)

dtype = np.float32
#tf_dtype = tf.float32
lambda_=3e-3
lr = 0.2
dsize = 2
fs = [dsize, 2, 2, 2]  # layer sizes

nonlin = tf.nn.sigmoid
def d_nonlin(y): return y*(1-y)


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

def regularized_inverse(mat):
  n = int(mat.shape[0])
  assert n == int(mat.shape[1])
  regmat = mat + lambda_*Identity(n)
  #  return mat
  return tf.constant(scipy.linalg.inv(regmat.numpy()))


global_cov_A = None
global_whitened_A = None
@profile
def loss_and_output_and_grad(Wf):
  """Returns cost, gradient for current parameter vector."""
  global fs, X, global_cov_A, global_whitened_A
  
  W = u.unflatten(Wf, fs[1:])   # perftodo: this creates transposes
  W.insert(0, X)

  A = [None]*(n+2)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = nonlin(W[i] @ A[i])
#    print(A[i+1])
#    #    print(A[i+1])
  err = (A[n+1] - A[1])

  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_nonlin(A[n+1])
  #  sampled_labels = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)

  #  print('random')
  #  print(np.random.randn(*X.shape).astype(dtype))
  noise = tf.constant(np.random.randn(*err.shape).astype(dtype))
#  print(noise)
  B2[n] = noise*d_nonlin(A[n+1])
#  print("B2[n]", B2[n])
#  print(B[n])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    backprop2 = t(W[i+1]) @ B2[i+1]
    B[i] = backprop*d_nonlin(A[i+1])
    B2[i] = backprop2*d_nonlin(A[i+1])

  dW = [None]*(n+1)
  pre_dW = [None]*(n+1)  # preconditioned dW

  cov_A = [None]*(n+1)    # covariance of activations[i]
  whitened_A = [None]*(n+1)    # covariance of activations[i]
  cov_B2 = [None]*(n+1)   # covariance of synthetic backprops[i]
  cov_B = [None]*(n+1)   # covariance of synthetic backprops[i]
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)

  if global_cov_A is None:
    global_cov_A = A[1]@t(A[1])/dsize
    global_whitened_A = regularized_inverse(global_cov_A) @ A[1]

  cov_A[1] = global_cov_A
  whitened_A[1] = global_whitened_A
    
  for i in range(1,n+1):
    if i > 1:
      cov_A[i] = A[i]@t(A[i])/dsize
      whitened_A[i] = regularized_inverse(cov_A[i]) @ A[i]
    cov_B2[i] = B2[i]@t(B2[i])/dsize
    cov_B[i] = B[i]@t(B[i])/dsize
    if SYNTHETIC_LABELS:
      whitened_B = regularized_inverse(cov_B2[i]) @ B[i]
    else:
      whitened_B = regularized_inverse(cov_B[i]) @ B[i]

    #regularized_inverse(cov_B[i])
    #    print("A", i, cov_A[i], regularized_inverse(cov_A[i]))
    #    print("B", i, cov_B[i], regularized_inverse(cov_B[i]))
    
    #    pre_dW[i] = (whitened_B @ t(whitened_A[i]))/dsize
    #    print(i, 'A', A[i].numpy())
    #    print(regularized_inverse(cov_A[i]).numpy())
    pre_dW[i] = (whitened_B @ t(whitened_A[i]))/dsize
    
    dW[i] = (B[i] @ t(A[i]))/dsize

  loss = u.L2(err)/2/dsize
  grad = u.flatten(dW[1:])
  kfac_grad = u.flatten(pre_dW[1:])
  return loss, A[n+1], grad, kfac_grad

def main():
  global fs, X, n, f, dsize, lambda_
  
  np.random.seed(1)
  tf.set_random_seed(1)
  
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2
  train_images = np.asarray([[0, 1], [2, 3]]).astype(dtype)
  X = tf.constant(train_images[:,:dsize].astype(dtype))


  W0_0 = np.asarray([[0., 1], [2, 3]]).astype(dtype)/10
  W1_0 = np.asarray([[4., 5], [6, 7]]).astype(dtype)/10
  W0f = u.flatten([W0_0, W1_0])
  Wf = tf.constant(W0f)

  losses = []
  for step in range(10):
    loss, output, grad, kfac_grad = loss_and_output_and_grad(Wf)
    loss0 = loss.numpy()
    print("Step %3d loss %10.9f"%(step, loss0))
    losses.append(loss0)

    Wf-=lr*kfac_grad
    u.record_time()

  u.summarize_time()
  target = 1.252017617  # without random sampling
  target = 1.256854534  # with random sampling but fixed seed
  target = 0.000359572  # with random sampling and linear 
  target = 1.251557469  # with random sampling

  assert abs(loss0-target)<1e-9, abs(loss0-target)

if __name__=='__main__':
  main()

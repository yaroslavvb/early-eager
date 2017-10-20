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

import common_gd
args = common_gd.args
args.cuda = not args.no_cuda and (tfe.num_gpus() > 0)

dtype = np.float32
tf_dtype = tf.float32
lambda_=3e-3
lr = 0.2
dsize = 2
fs = [dsize, 2, 2, 2]  # layer sizes

#nonlin = tf.identity
#def d_nonlin(y): return 1
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


@tfe.custom_gradient
def regular_matmul(x, y):
  result = x @ y
  def grad(dr):
    return [dr @ tf.transpose(y), tf.transpose(x) @ dr]
  return result, grad

@tfe.custom_gradient
def capturing_matmul(x, y):
  global forward, backward
  result = x @ y
  forward.append(y)
  def grad(dr):
    backward.append(dr)
    return [dr @ tf.transpose(y), tf.transpose(x) @ dr]
  return result, grad

@tfe.custom_gradient
def kfac_matmul(W, A):
  def grad(B):
    kfac_A = forward_inv.pop() @ A
    kfac_B = backward_inv.pop() @ B
    return [kfac_B @ tf.transpose(kfac_A), tf.transpose(W) @ B]
  
  return W @ A, grad

forward = []
backward = []
def main():
  global fs, X, n, f, dsize, lambda_, forward, backward, forward_inv, backward_inv, idx
  matmul = regular_matmul
  
  np.random.seed(args.seed)
  tf.set_random_seed(args.seed)
  
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2
  train_images = np.asarray([[0, 1], [2, 3]]).astype(dtype)
  X = tf.constant(train_images[:,:dsize].astype(dtype))


  W1_0 = np.asarray([[0., 1], [2, 3]]).astype(dtype)/10
  W2_0 = np.asarray([[4., 5], [6, 7]]).astype(dtype)/10
  W0f = u.flatten([W1_0, W2_0])
  Wf = tf.constant(W0f)

  W1 = tfe.Variable(W1_0, name='W1')
  W2 = tfe.Variable(W2_0, name='W2')

  def loss_fn(target):
    x = nonlin(capturing_matmul(W1, X))
    x = nonlin(capturing_matmul(W2, x))
    err = target-x
    loss = tf.reduce_sum(err*err)/2/dsize
    return loss

  def kfac_loss_fn(target):
    x = nonlin(kfac_matmul(W1, X))
    x = nonlin(kfac_matmul(W2, x))
    err = target-x
    loss = tf.reduce_sum(err*err)/2/dsize
    return loss


  forward_inv = [tf.zeros((2, 2))]*n
  backward_inv = [tf.zeros((2, 2))]*n
  capturing_value_and_gradients_fn = tfe.implicit_value_and_gradients(loss_fn)
  kfac_value_and_gradients_fn = tfe.implicit_value_and_gradients(kfac_loss_fn)

  losses = []
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

  for step in range(10):
    backward = []
    forward = []
    loss, _ = capturing_value_and_gradients_fn(X)
    loss0 = loss.numpy()
    print("Step %3d loss %10.9f"%(step, loss0))

    # adjust to have same scale
    for i in range(len(backward)):
      backward[i] = dsize*backward[i]
    backward = backward[::-1]

    def cov(X):
      return X @ t(X)/dsize
    def invcov(X):
      cov = X @ t(X)/dsize
      return regularized_inverse(cov)
      
    def invert_covariances(X_list):
      result = []
      for X in X_list:
        cov = X @ t(X)/dsize
        result.append(regularized_inverse(cov))
      return result

    forward_inv = invert_covariances(forward)
    backward_inv = invert_covariances(backward)

    idx = 0
    loss, grads_and_vars = kfac_value_and_gradients_fn(X)
    losses.append(loss0)
    optimizer.apply_gradients(grads_and_vars)


    u.record_time()

  u.summarize_time()
  target = 1.251557469
  target = 1.647592664 # switching to gd
  target = 0.663335681 # switching to linear activation
  target = 1.994378567 # after switching to 2 steps
  target = 1.252017617 # switching to kfac no sampling
  assert abs(loss0-target)<1e-9, abs(loss0-target)

if __name__=='__main__':
  main()

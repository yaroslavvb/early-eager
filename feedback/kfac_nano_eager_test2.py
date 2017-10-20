import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

# for line profiling
try:
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.

dtype = np.float32
lambda_=3e-3
lr = 0.2
dsize = 2
fs = [dsize, 2, 2, 2]  # layer sizes
n = len(fs) - 2
nonlin = tf.nn.sigmoid


def t(mat): return tf.transpose(mat)

def regularized_inverse(mat):
  n = int(mat.shape[0])
  return tf.linalg.inv(mat + lambda_*tf.eye(n))


forward = []
backward = []
forward_inv = []
backward_inv = []

@tfe.custom_gradient
def capturing_matmul(W, A):
  forward.append(A)
  def grad(B):
    backward.append(B)
    return [B @ tf.transpose(A), tf.transpose(W) @ B]
  return W @ A, grad


@tfe.custom_gradient
def kfac_matmul(W, A):
  def grad(B):
    kfac_A = forward_inv.pop() @ A
    kfac_B = backward_inv.pop() @ B
    return [kfac_B @ tf.transpose(kfac_A), tf.transpose(W) @ B]
  return W @ A, grad


def main():
  np.random.seed(1)
  tf.set_random_seed(1)
  global kfac_matmul
  
  train_images = np.asarray([[0, 1], [2, 3]])
  X = tf.constant(train_images[:,:dsize].astype(dtype))

  W1_0 = np.asarray([[0., 1], [2, 3]]).astype(dtype)/10
  W2_0 = np.asarray([[4., 5], [6, 7]]).astype(dtype)/10
  W1 = tfe.Variable(W1_0, name='W1')
  W2 = tfe.Variable(W2_0, name='W2')

  def general_loss_fn(target):
    x = nonlin(matmul(W1, X))
    x = nonlin(matmul(W2, x))
    err = target-x
    loss = tf.reduce_sum(err*err)/2/dsize
    return loss
  
  def capture_loss_fn(target):
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

  capture_fn = tfe.implicit_value_and_gradients(capture_loss_fn)
  kfac_fn = tfe.implicit_value_and_gradients(kfac_loss_fn)
  general_fn = tfe.implicit_value_and_gradients(general_loss_fn)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

  for step in range(10):
    del backward[:]
    del forward[:]
    del forward_inv[:]
    del backward_inv[:]

    loss, _ = capture_fn(X)
    backward.reverse()
    print("Step %3d loss %10.9f"%(step, loss.numpy()))

    def cov(X): return X @ t(X)/dsize
    def invcov(X): return regularized_inverse(cov(X))
      
    for i in range(n):
      # divide by dsize^2 to keep learning rates compatible with SGD
      forward_inv.append(invcov(forward[i]))
      backward_inv.append(invcov(backward[i])/dsize/dsize)

    _, grads_and_vars = kfac_fn(X)
    #    kfac_matmul = capturing_matmul
    optimizer.apply_gradients(grads_and_vars)

  target = 1.259010077
  assert abs(loss.numpy()-target)<1e-9, abs(loss.numpy()-target)

if __name__=='__main__':
  main()

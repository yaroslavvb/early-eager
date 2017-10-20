import numpy as np
import tensorflow as tf
import scipy
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()


def main():
  np.random.seed(1)
  tf.set_random_seed(2)
  
  dtype = np.float32
  lambda_=3e-3
  lr = 0.2
  dsize = 2

  def t(mat): return tf.transpose(mat)

  def regularized_inverse(mat):
    n = int(mat.shape[0])
    return tf.linalg.inv(mat + lambda_*tf.eye(n, dtype=dtype))

  train_images = np.asarray([[0, 1], [2, 3]])
  X = tf.constant(train_images[:,:dsize].astype(dtype))

  W1_0 = np.asarray([[0., 1], [2, 3]]).astype(dtype)/10
  W2_0 = np.asarray([[4., 5], [6, 7]]).astype(dtype)/10
  W1 = tfe.Variable(W1_0, name='W1')
  W2 = tfe.Variable(W2_0, name='W2')

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

  matmul = tf.matmul
  def loss_fn(synthetic=False):
    x = tf.nn.sigmoid(matmul(W1, X))
    x = tf.nn.sigmoid(matmul(W2, x))
    if synthetic:
      noise = tf.random_normal(X.shape)
      target = tf.constant((x + noise).numpy())
    else:
      target = X
    err = target-x
    loss = tf.reduce_sum(err*err)/2/dsize
    return loss
  
  loss_and_grads = tfe.implicit_value_and_gradients(loss_fn)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
  for step in range(10):
    del backward[:]
    del forward[:]
    del forward_inv[:]
    del backward_inv[:]

    matmul = capturing_matmul
    loss, grads_and_vars = loss_and_grads(True)
    backward.reverse()

    for i in range(len(backward)):
      backward[i] = backward[i]*dsize
      
    def cov(X): return X @ t(X)/dsize
    def invcov(X): return regularized_inverse(cov(X))
      
    for i in range(2):
      forward_inv.append(invcov(forward[i]))
      backward_inv.append(invcov(backward[i]))

    matmul = kfac_matmul
    loss, grads_and_vars = loss_and_grads()
    print("Step %3d loss %10.9f"%(step, loss.numpy()))
    optimizer.apply_gradients(grads_and_vars)

  target = 1.251444697  # with proper random sampling
  assert abs(loss.numpy()-target)<1e-9, abs(loss.numpy()-target)

if __name__=='__main__':
  main()

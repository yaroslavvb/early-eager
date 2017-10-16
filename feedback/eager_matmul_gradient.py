import tensorflow as tf
import numpy as np
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

@tfe.custom_gradient
def my_matmul(x, y):
  result = x @ y
  def grad(dr):
    return [dr @ tf.transpose(y), tf.transpose(x) @ dr]
  return result, grad

lr = 0.25
n = 2
x = tfe.Variable(tf.ones((n, n)), name="x")
y = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

def loss_fn(x): return tf.reduce_sum(my_matmul(x, y))
loss_grads_fn = tfe.value_and_gradients_function(loss_fn)

for step in range(5):
  loss, grads = loss_grads_fn(x)
  print("loss =", loss.numpy())
  x.assign_sub(lr*grads[0])

assert loss.numpy() == -96

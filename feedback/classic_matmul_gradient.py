import tensorflow as tf
from tensorflow.python.framework import function

dtype=tf.float32
@function.Defun(dtype, dtype, dtype)
def my_matmul_grad(x, y, dr):
  return [dr @ tf.transpose(y), tf.transpose(x) @ dr]

@function.Defun(dtype, dtype, grad_func=my_matmul_grad)
def my_matmul(x, y):
  return x@y

def main():
  n = 2
  lr = 0.25
  x = tf.Variable(tf.ones((n, n)))
  y = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
  loss = tf.reduce_sum(my_matmul(x, y))
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  train_op = opt.minimize(loss)

  sess = tf.InteractiveSession()
  dtype = tf.float32
  sess.run(x.initializer)
  
  for i in range(5):
    loss0 = loss.eval()
    print("loss =", loss0)
    train_op.run()
  assert loss0 == -96
  
if __name__=='__main__':
  main()

import util as u

import tensorflow as tf
import numpy as np
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

import common_gd
args = common_gd.args
args.cuda = not args.no_cuda and (tfe.num_gpus() > 0)

def main():
  tf.set_random_seed(args.seed)
  np.random.seed(args.seed)
  
  images = tf.constant(u.get_mnist_images().T)
  images = images[:args.batch_size]
  if args.cuda:
    images = images.as_gpu_tensor()
  data = images

  if args.cuda:
    device='/gpu:0'
  else:
    device=''

  with tf.device(device):
    encoder = tf.layers.Dense(units=args.hidden_size, use_bias=False,
                            activation=tf.sigmoid)
    decoder = tf.layers.Dense(units=args.visible_size, use_bias=False,
                              activation=tf.sigmoid)
    def loss_fn(inputs):
      predictions = decoder(encoder(inputs))
      return tf.reduce_mean(tf.square(predictions-inputs))
    value_and_gradients_fn = tfe.implicit_value_and_gradients(loss_fn)

    # initialize weights
    loss_fn(data)
    params1 = encoder.weights[0]
    params2 = decoder.weights[0]
    params1.assign(u.ng_init(args.visible_size, args.hidden_size))
    params2.assign(u.ng_init(args.hidden_size, args.visible_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
    for step in range(args.iters):
      value, grads_and_vars = value_and_gradients_fn(data)
      optimizer.apply_gradients(grads_and_vars)

      print("Step %3d loss %6.5f"%(step, value.numpy()))
      u.record_time()

    u.summarize_time()
    

if __name__=='__main__':
  main()

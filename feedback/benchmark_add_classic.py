import argparse
parser = argparse.ArgumentParser(description='benchmark')

parser.add_argument('--iters', type=int, default=100, metavar='N',
                    help='number of iterations to run for (default: 20)')
parser.add_argument('--size_gb', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
args = parser.parse_args()

import tensorflow as tf
import numpy as np
import time

#from tensorflow.contrib.eager.python import tfe
#tfe.enable_eager_execution()

def benchmark():
  dtype = np.float32
  
  device_ctx = tf.device('/gpu:0')
  device_ctx.__enter__()

  SIZE_GB=args.size_gb
  n = int(SIZE_GB*1000*250*1000)
  infeed = tf.placeholder(dtype, shape=(n,))

  # turn off optimizations just in case
  sess = tf.Session(config=config)
  a = tf.Variable(infeed)
  sess.run(a.initializer, feed_dict={infeed: np.ones((n,),dtype=dtype)})
  b = a
  times = []
  iters = args.iters
  for i in range(iters):
    b+=a
  sess.run(b.op)
  
  sess.run(a.initializer, feed_dict={infeed: np.ones((n,),dtype=dtype)})
  time0 = time.perf_counter()
  result = sess.run(b[0])
  elapsed_ms = (time.perf_counter()-time0)*1000
  print('result after %d iters: %d' % (iters, result,))

  print("%d additions of %.3f GB in %.3f ms (%.3f per addition)"%(iters,
                                                                  SIZE_GB,
                                                                  elapsed_ms,
                                                                  elapsed_ms/iters))

if __name__=='__main__':
  benchmark()

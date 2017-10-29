# result after 100 iters: 101
# 100 additions of 1.000 GB in 2520.112 ms (25.201 per addition)

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

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

def benchmark():
  global a, b
  SIZE_GB=args.size_gb
  iters = args.iters
  
  dtype = np.float32
  device_ctx = tf.device('/gpu:0')
  device_ctx.__enter__()

  iters = 100
  def compute():
    global a, b
    times = []
    for i in range(iters):
      b+=a
    return b

  a = tf.ones(int(SIZE_GB*1000*250000), dtype=dtype)
  b = a
  compute()
  a = tf.ones(int(SIZE_GB*1000*250000), dtype=dtype)
  b = a
  
  start = time.perf_counter()
  result = compute()[0].numpy()
  elapsed_ms = (time.perf_counter()-start)*1000


  print('result after %d iters: %d' % (iters, result,))
  print("%d additions of %.3f GB in %.3f ms (%.3f per addition)"%(iters,
                                                                  SIZE_GB,
                                                                  elapsed_ms,
                                                                  elapsed_ms/iters))
  


if __name__=='__main__':
  benchmark()

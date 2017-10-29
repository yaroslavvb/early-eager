# result after 100 iters: 101
# 100 additions of 1.000 GB in 1447.075 ms (14.471 per addition)
#
# Note, call .cuda() before measuring, 12GB/sec transfer = 160ms to copy things

import argparse
parser = argparse.ArgumentParser(description='benchmark')

parser.add_argument('--iters', type=int, default=100, metavar='N',
                    help='number of iterations to run for (default: 20)')
parser.add_argument('--size_gb', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
args = parser.parse_args()

import torch
import numpy as np
import time

def benchmark():
  global a, b
  SIZE_GB=args.size_gb
  iters = args.iters
  a = torch.ones(int(SIZE_GB*1000*250000)).cuda()
  b = torch.ones(int(SIZE_GB*1000*250000)).cuda()
  iters = args.iters
  def compute():
    global a, b
    for i in range(iters):
      b+=a
    return b
  compute()  # pre-warm
  a = torch.ones(int(SIZE_GB*1000*250000)).cuda()
  b = torch.ones(int(SIZE_GB*1000*250000)).cuda()
  start = time.perf_counter()
  result = compute()[0]
  elapsed_ms = (time.perf_counter()-start)*1000

  print('result after %d iters: %d' % (iters, result,))

  print("%d additions of %.3f GB in %.3f ms (%.3f per addition)"%(iters,
                                                                  SIZE_GB,
                                                                  elapsed_ms,
                                                                  elapsed_ms/iters))


if __name__=='__main__':
  benchmark()

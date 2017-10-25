import tensorflow as tf
import eager_lbfgs
import pytorch_lbfgs

import time

def main():
  iters = 11

  #batch_sizes = [1, 10, 100, 1000, 10000, 60000]
  batch_sizes = [10]

  eager_stats = []
  pytorch_stats = []

  def benchmark(f):
    # do whole run once for pre-warming
    f()  
    start_time = time.perf_counter()
    final_loss = f()
    elapsed_time = time.perf_counter() - start_time
    return final_loss, elapsed_time

  for batch_size in batch_sizes:
    print('running eager')
    def eager_run():
      return eager_lbfgs.benchmark(batch_size=batch_size, max_iter=iters)
    eager_stats.append(benchmark(eager_run))

    print('running pytorch')
    def pytorch_run():
      return pytorch_lbfgs.benchmark(batch_size=batch_size, max_iter=iters)
    pytorch_stats.append(benchmark(pytorch_run))
    
  print(eager_stats)
  print(pytorch_stats)
    
if __name__=='__main__':
  main()

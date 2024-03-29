#!/usr/bin/env python
GLOBAL_PROFILE = False   # do timelines

import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'  # autotune adds random memory spikes
# hack to work around broken initializer
from tensorflow.python.ops import variables
def passthrough(obj, value): return value
try:
  variables.Variable._build_initializer_expr=passthrough
except: # older versions of TF don't have this
  pass

from tensorflow.python.client import timeline
import argparse
import json
import sys
import time
import util as u
import util
from util import t  # transpose

use_tikhonov=False

# Test generation releases
release_name='mnist_deep'   # deep synchronous MNIST (oct_batches12)
release_test_fn = release_name+'_losses_test.csv'

# for line profiling
try:
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.

  
#import load_MNIST

import kfac as kfac_lib
from kfac import Model
from kfac import Kfac
from kfac import IndexedGrad
import kfac

import sys
import tensorflow as tf
import numpy as np


# TODO: get rid of this
#purely_linear = False  # convert sigmoids into linear nonlinearities
#purely_relu = True     # convert sigmoids into ReLUs


# TODO: get rid
def W_uniform(s1, s2): # uniform weight init from Ng UFLDL
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result


def ng_init(rows, cols):
  # creates uniform initializer using Ng's formula
  # TODO: turn into TF
  r = np.sqrt(6) / np.sqrt(rows + cols + 1)
  result = np.random.random(rows*cols)*2*r-r
  return result.reshape((rows, cols))


full_trace_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                                   output_partition_graphs=True)
def sessrun(*args, **kwargs):
  sess = u.get_default_session()
  if not GLOBAL_PROFILE:
    return sess.run(*args, **kwargs)
  
  run_metadata = tf.RunMetadata()

  kwargs['options'] = full_trace_options
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)
  first_entry = args[0]
  if isinstance(first_entry, list):
    if len(first_entry) == 0 and len(args) == 1:
      return None
    first_entry = first_entry[0]
  name = first_entry.name
  name = name.replace('/', '-')

  tl = timeline.Timeline(run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format()
  with open('timelines/%s.json'%(name,), 'w') as f:
    f.write(ctf)
  with open('timelines/%s.pbtxt'%(name,), 'w') as f:
    f.write(str(run_metadata))
  return result


def model_creator(batch_size, name="default", dtype=np.float32):
  """Create MNIST autoencoder model. Dataset is part of model."""

  model = Model(name)

  def get_batch_size(data):
    return 10000

  init_dict = {}
  global_vars = []
  local_vars = []
  
  # TODO: factor out to reuse between scripts
  # TODO: change feed_dict logic to reuse value provided to VarStruct
  # current situation makes reinitialization of global variable change
  # it's value, counterinituitive
  def init_var(val, name, is_global=False):
    """Helper to create variables with numpy or TF initial values."""
    if isinstance(val, tf.Tensor):
      var = u.get_variable(name=name, initializer=val, reuse=is_global)
    else:
      val = np.array(val)
      assert u.is_numeric(val), "Non-numeric type."
      
      var_struct = u.get_var(name=name, initializer=val, reuse=is_global)
      holder = var_struct.val_
      init_dict[holder] = val
      var = var_struct.var

    if is_global:
      global_vars.append(var)
    else:
      local_vars.append(var)
      
    return var

  def nonlin(x):
    return tf.sigmoid(x)

  # TODO: rename into "nonlin_d"
  def d_nonlin(y):
    return y*(1-y)

  patches = train_images[:,:args.batch_size];
  test_patches = test_images[:,:args.batch_size];

  if args.dataset == 'cifar':
    input_dim = 3*32*32
  elif args.dataset == 'mnist':
    input_dim = 28*28
  else:
    assert False
  if release_name == 'kfac_tiny':
    fs = [args.batch_size, input_dim, 196, input_dim]
  else:
    fs = [args.batch_size, input_dim, 1024, 1024, 1024, 196, 1024, 1024, 1024,
          input_dim]
    
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2

  # Full dataset from which new batches are sampled
  X_full = init_var(train_images, "X_full", is_global=True)

  X = init_var(patches, "X", is_global=False)  # stores local batch per model
  W = [None]*n
  W.insert(0, X)
  A = [None]*(n+2)
  A[1] = W[0]
  for i in range(1, n+1):
    init_val = ng_init(f(i), f(i-1)).astype(dtype)
    W[i] = init_var(init_val, "W_%d"%(i,), is_global=True)
    A[i+1] = nonlin(kfac_lib.matmul(W[i], A[i]))
  err = A[n+1] - A[1]
  model.loss = u.L2(err) / (2 * get_batch_size(err))

  # create test error eval
  layer0 = init_var(test_patches, "X_test", is_global=True)
  layer = layer0
  for i in range(1, n+1):
    layer = nonlin(W[i] @ layer)
  verr = (layer - layer0)
  model.vloss = u.L2(verr) / (2 * get_batch_size(verr))

  # manually compute backprop to use for sanity checking
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_nonlin(A[n+1])
  _sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  if args.fixed_labels:
    _sampled_labels_live = tf.ones(shape=(f(n), f(-1)), dtype=dtype)
    
  _sampled_labels = init_var(_sampled_labels_live, "to_be_deleted",
                             is_global=False)

  B2[n] = _sampled_labels*d_nonlin(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    B[i] = backprop*d_nonlin(A[i+1])
    backprop2 = t(W[i+1]) @ B2[i+1]
    B2[i] = backprop2*d_nonlin(A[i+1])

  # cov_A = [None]*(n+1)    # covariance of activations[i]
  # cov_B2 = [None]*(n+1)   # covariance of synthetic backprops[i]
#  vars_svd_A = [None]*(n+1)
#  vars_svd_B2 = [None]*(n+1)
#  dW = [None]*(n+1)
#  pre_dW = [None]*(n+1)   # preconditioned dW
  # todo: decouple initial value from covariance update
  # # maybe need start with identity and do running average
  # for i in range(1,n+1):
  #   if regularized_svd:
  #     cov_A[i] = init_var(A[i]@t(A[i])/args.batch_size+args.Lambda*u.Identity(f(i-1)), "cov_A%d"%(i,))
  #     cov_B2[i] = init_var(B2[i]@t(B2[i])/args.batch_size+args.Lambda*u.Identity(f(i)), "cov_B2%d"%(i,))
  #   else:
  #     cov_A[i] = init_var(A[i]@t(A[i])/args.batch_size, "cov_A%d"%(i,))
  #     cov_B2[i] = init_var(B2[i]@t(B2[i])/args.batch_size, "cov_B2%d"%(i,))
#    vars_svd_A[i] = u.SvdWrapper(cov_A[i],"svd_A_%d"%(i,), do_inverses=False)
#    vars_svd_B2[i] = u.SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,), do_inverses=False)
    
#    whitened_A = u.cached_inverse(vars_svd_A[i], args.Lambda) @ A[i]
#    whitened_B = u.cached_inverse(vars_svd_B2[i], args.Lambda) @ B[i]
#    dW[i] = (B[i] @ t(A[i]))/args.batch_size
#    pre_dW[i] = (whitened_B @ t(whitened_A))/args.batch_size

    
  sampled_labels_live = A[n+1] + tf.random_normal((f(n), f(-1)),
                                                  dtype=dtype, seed=0)
  if args.fixed_labels:
    sampled_labels_live = A[n+1]+tf.ones(shape=(f(n), f(-1)), dtype=dtype)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels", is_global=False)
  err2 = A[n+1] - sampled_labels
  model.loss2 = u.L2(err2) / (2 * args.batch_size)
  model.global_vars = global_vars
  model.local_vars = local_vars
  model.trainable_vars = W[1:]

  # todo, we have 3 places where model step is tracked, reduce
  model.step = init_var(u.as_int32(0), "step", is_global=False)
  advance_step_op = model.step.assign_add(1)
  assert get_batch_size(X_full) % args.batch_size == 0
  batches_per_dataset = (get_batch_size(X_full) // args.batch_size)
  batch_idx = tf.mod(model.step, batches_per_dataset)
  start_idx = batch_idx * args.batch_size
  advance_batch_op = X.assign(X_full[:,start_idx:start_idx + args.batch_size])
  
  def advance_batch():
    #    print("Step for model(%s) is %s"%(model.name, u.eval(model.step)))
    sess = u.get_default_session()
    # TODO: get rid of _sampled_labels
    sessrun([sampled_labels.initializer, _sampled_labels.initializer])
    if args.advance_batch:
      sessrun(advance_batch_op)
    sessrun(advance_step_op)
    
  model.advance_batch = advance_batch

  # TODO: refactor this to take initial values out of Var struct
  #global_init_op = tf.group(*[v.initializer for v in global_vars])
  global_init_ops = [v.initializer for v in global_vars]
  global_init_op = tf.group(*[v.initializer for v in global_vars])
  global_init_query_ops = [tf.logical_not(tf.is_variable_initialized(v))
                           for v in global_vars]
  
  def initialize_global_vars(verbose=False, reinitialize=False):
    """If reinitialize is false, will not reinitialize variables already
    initialized."""
    
    sess = u.get_default_session()
    if not reinitialize:
      uninited = sessrun(global_init_query_ops)
      # use numpy boolean indexing to select list of initializers to run
      to_initialize = list(np.asarray(global_init_ops)[uninited])
    else:
      to_initialize = global_init_ops
      
    if verbose:
      print("Initializing following:")
      for v in to_initialize:
        print("   " + v.name)

    sessrun(to_initialize, feed_dict=init_dict)
  model.initialize_global_vars = initialize_global_vars

  # didn't quite work (can't initialize var in same run call as deps likely)
  # enforce that batch is initialized before everything
  # except fake labels opa
  # for v in local_vars:
  #   if v != X and v != sampled_labels and v != _sampled_labels:
  #     print("Adding dep %s on %s"%(v.initializer.name, X.initializer.name))
  #     u.add_dep(v.initializer, on_op=X.initializer)
      
  local_init_op = tf.group(*[v.initializer for v in local_vars],
                           name="%s_localinit"%(model.name))
  print("Local vars:")
  for v in local_vars:
    print(v.name)
    
  def initialize_local_vars():
    sess = u.get_default_session()
    sessrun(_sampled_labels.initializer, feed_dict=init_dict)
    sessrun(X.initializer, feed_dict=init_dict)
    sessrun(local_init_op, feed_dict=init_dict)
  model.initialize_local_vars = initialize_local_vars

  return model

@profile
def main():
  np.random.seed(1)
  tf.set_random_seed(1)

  logger = u.TensorboardLogger(args.run)
  
  with u.timeit("init/session"):

    rewrite_options=None
    try:
      from tensorflow.core.protobuf import rewriter_config_pb2
      rewrite_options = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True,
        constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
        memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
    except:
      pass
      
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options,
                                  rewrite_options=rewrite_options)
    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(graph_options=graph_options,
                            gpu_options=gpu_options,
                            log_device_placement=False)

    sess = tf.InteractiveSession(config=config)
    u.register_default_session(sess)   # since default session is Thread-local

  with u.timeit("init/model_init"):
    model = model_creator(args.batch_size, name="main")
    model.initialize_global_vars(verbose=True)
    model.initialize_local_vars()

  kfac_lib.numeric_inverse = args.numeric_inverse
  with u.timeit("init/kfac_init"):
    kfac = Kfac(model_creator, args.kfac_batch_size) 
    kfac.model.initialize_global_vars(verbose=False)
    kfac.model.initialize_local_vars()
    kfac.Lambda.set(args.Lambda)
    kfac.reset()    # resets optimization variables (not model variables)

  if args.mode != 'run':
    opt = tf.train.AdamOptimizer(0.001)
  else:
    #    opt = tf.train.AdamOptimizer(args.lr)
    opt = tf.train.GradientDescentOptimizer(args.lr)
  grads_and_vars = opt.compute_gradients(model.loss,
                                         var_list=model.trainable_vars)
    
      
  grad = IndexedGrad.from_grads_and_vars(grads_and_vars)
  grad_new = kfac.correct(grad)
  with u.capture_vars() as adam_vars:
    train_op = opt.apply_gradients(grad_new.to_grads_and_vars())
  with u.timeit("init/adam"):
    sessrun([v.initializer for v in adam_vars])
  
  losses = []
  u.record_time()

  start_time = time.time()
  vloss0 = 0

  # todo, unify the two data outputs
  outfn = 'data/%s_%f_%f.csv'%(args.run, args.lr, args.Lambda)

  start_time = time.time()
  if args.extra_kfac_batch_advance:
    kfac.model.advance_batch()  # advance kfac batch

  if args.kfac_async:
    kfac.start_stats_runners()
    
  for step in range(args.num_steps):
    vloss0 = 0
    if args.validate_every_n and step%args.validate_every_n == 0:
      loss0 = sessrun(model.loss)
    else:
      loss0, = sessrun([model.loss])
    losses.append(loss0)  # TODO: remove this

    logger('loss/loss', loss0, 'loss/vloss', vloss0)
    
    elapsed = time.time()-start_time
    start_time = time.time()
    print("%4d ms, step %4d, loss %10.9f, vloss %5.2f" %(elapsed*1e3, step,
                                                        loss0, vloss0))

    if args.method=='kfac' and not args.kfac_async:
      kfac.model.advance_batch()
      kfac.update_stats()

    with u.timeit("train"):
      model.advance_batch()
      with u.timeit("grad.update"):
        grad.update()
      with kfac.read_lock():
        grad_new.update()
      u.run(train_op)
      u.record_time()

    logger.next_step()

  # TODO: use u.global_runs_dir
  # TODO: get rid of u.timeit?
  
  #  with open('timelines/graphdef.txt', 'w') as f:
  #    f.write(str(u.get_default_graph().as_graph_def()))
  u.summarize_time()
  

  target = 40.707656860
  assert abs(loss0-target)<1e-9, abs(loss0-target)
  assert u.last_time()<800, "Expected 648 on GTX 1080"

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, default='run', help='record to record test data, test to perform test, run to run training for longer')
  parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
  parser.add_argument('--method', type=str, default="kfac", help='turn on KFAC')
  parser.add_argument('--fixed_labels', type=int, default=0,
                      help='if true, fix synthetic labels to all 1s')
  parser.add_argument('--lr', type=float, default=0.001,
                      help='learning rate to use')
  parser.add_argument('--validate_every_n', type=int, default=10,
                      help='set to positive number to measure validation')
  # lambda tuning graphs: https://wolfr.am/lojcyhYz
  parser.add_argument('-L', '--Lambda', type=float, default=0.01,
                      help='lambda value')
  parser.add_argument('-r', '--run', type=str, default='default',
                      help='name of experiment run')
  parser.add_argument('-n', '--num_steps', type=int, default=1000000,
                      help='number of steps')
  parser.add_argument('--dataset', type=str, default="cifar",
                      help='which dataset to use')
  # todo: split between optimizer batch size and stats batch size
  parser.add_argument('-b', '--batch_size', type=int, default=10000,
                      help='batch size')
  parser.add_argument('--kfac_batch_size', type=int, default=10000,
                      help='batch size to use for KFAC stats')
  parser.add_argument('--dataset_size', type=int, default=1000000000,
                      help='truncate dataset at this value')
  parser.add_argument('--advance_batch', type=int, default=0,
                      help='whether to advance batch')
  parser.add_argument('--extra_kfac_batch_advance', type=int, default=0,
                      help='make kfac batches out of sync')
  parser.add_argument('--kfac_polyak_factor', type=float, default=1.0,
                      help='polyak averaging factor to use')
  parser.add_argument('--kfac_async', type=int, default=0,
                      help='do covariance and inverses asynchronously')
  parser.add_argument('-nr', '--numeric_inverse', type=int, default=0,
                      help='estimate inverse numerically')

  args = parser.parse_args()
  u.set_global_args(args)
  print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

  args.num_steps = 10
  args.advance_batch = 0
  args.extra_kfac_batch_advance = 0
  dsize = 10000
  args.batch_size = dsize
  args.dataset = 'mnist'
  args.lr = 0.2
      
  train_images = u.get_mnist_images('train')
  test_images = u.get_mnist_images('test')
  train_images = train_images[:,:dsize]  # batch last

  main()

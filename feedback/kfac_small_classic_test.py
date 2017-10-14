import util as u
from util import t  # transpose
u.check_mkl()

import numpy as np
import scipy
import tensorflow as tf
import time


# fix for https://github.com/tensorflow/tensorflow/issues/13351
from tensorflow.python.ops import variables
def passthrough(obj, value): return value
try:
  variables.Variable._build_initializer_expr=passthrough
except: # older versions of TF don't have this
  pass

# for line profiling
try:
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.



@profile
def main():
  np.random.seed(0)
  tf.set_random_seed(0)
  
  dtype = np.float32

  train_images = u.get_mnist_images()
  
  dsize = 10000
  patches = train_images[:,:dsize].astype(dtype);
  fs = [dsize, 28*28, 196, 28*28]


  # values from deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
  X0=patches
  lambda_=3e-3
  rho=tf.constant(0.1, dtype=dtype)
  beta=3
  W0_0 = u.ng_init(fs[2],fs[3])
  W1_0 = u.ng_init(fs[3], fs[2])
  W0f = u.flatten([W0_0.flatten(), W1_0.flatten()])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  # helper to create variables with numpy or TF initial value
  init_dict = {}     # {var_placeholder: init_value}
  vard = {}          # {var: u.VarInfo}
  def init_var(val, name, trainable=False, noinit=False):
    if isinstance(val, tf.Tensor):
      collections = [] if noinit else None
      var = tf.Variable(val, name=name, collections=collections)
    else:
      val = np.array(val)
      assert u.is_numeric, "Unknown type"
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      var = tf.Variable(holder, name=name, trainable=trainable)
      init_dict[holder] = val
    var_p = tf.placeholder(var.dtype, var.shape)
    var_setter = var.assign(var_p)
    vard[var] = u.VarInfo(var_setter, var_p)
    return var

  lr = init_var(0.2, "lr")
    
  Wf = init_var(W0f, "Wf", True)
  Wf_copy = init_var(W0f, "Wf_copy", True)
  W = u.unflatten(Wf, fs[1:])   # perftodo: this creates transposes
  X = init_var(X0, "X")
  W.insert(0, X)

  def sigmoid(x):
    return tf.sigmoid(x)
      
  def d_sigmoid(y):
    return y*(1-y)
    
  def kl(x, y):
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))
  def d_kl(x, y):
    return (1-x)/(1-y) - x/y
  
  # A[i] = activations needed to compute gradient of W[i]
  # A[n+1] = network output
  A = [None]*(n+2)

  fail_node = tf.Print(0, [0], "fail, this must never run")
  with tf.control_dependencies([fail_node]):
    A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    
  # reconstruction error and sparsity error
  err = (A[3] - A[1])
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  # B2[i] = backprops from sampled labels needed for natural gradient
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels", noinit=True)
  B2[n] = sampled_labels*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    backprop2 = t(W[i+1]) @ B2[i+1]
    B[i] = backprop*d_sigmoid(A[i+1])
    B2[i] = backprop2*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  pre_dW = [None]*(n+1)  # preconditioned dW
  pre_dW_stable = [None]*(n+1)  # preconditioned stable dW

  cov_A = [None]*(n+1)    # covariance of activations[i]
  cov_B2 = [None]*(n+1)   # covariance of synthetic backprops[i]
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)
  for i in range(1,n+1):
    cov_op = A[i]@t(A[i])/dsize + lambda_*u.Identity(A[i].shape[0])
    cov_A[i] = init_var(cov_op, "cov_A%d"%(i,))
    cov_op = B2[i]@t(B2[i])/dsize + lambda_*u.Identity(B2[i].shape[0])
    cov_B2[i] = init_var(cov_op, "cov_B2%d"%(i,))
    vars_svd_A[i] = u.SvdWrapper(cov_A[i],"svd_A_%d"%(i,), do_inverses=True)
    vars_svd_B2[i] = u.SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,), do_inverses=True)
    whitened_A = vars_svd_A[i].inv @ A[i]
    whitened_B = vars_svd_B2[i].inv @ B[i]
    pre_dW[i] = (whitened_B @ t(whitened_A))/dsize
    dW[i] = (B[i] @ t(A[i]))/dsize

  # Loss function
  reconstruction = u.L2(err) / (2 * dsize)

  loss = reconstruction

  grad_live = u.flatten(dW[1:])
  pre_grad_live = u.flatten(pre_dW[1:]) # fisher preconditioned gradient
  grad = init_var(grad_live, "grad")
  pre_grad = init_var(pre_grad_live, "pre_grad")

  update_params_op = Wf.assign(Wf-lr*pre_grad).op
  save_params_op = Wf_copy.assign(Wf).op
  pre_grad_dot_grad = tf.reduce_sum(pre_grad*grad)
  grad_norm = tf.reduce_sum(grad*grad)
  pre_grad_norm = u.L2(pre_grad)

  def dump_svd_info(step):
    """Dump singular values and gradient values in those coordinates."""
    for i in range(1, n+1):
      svd = vars_svd_A[i]
      s0, u0, v0 = sess.run([svd.s, svd.u, svd.v])
      u.dump(s0, "A_%d_%d"%(i, step))
      A0 = A[i].eval()
      At0 = v0.T @ A0
      u.dump(A0 @ A0.T, "Acov_%d_%d"%(i, step))
      u.dump(At0 @ At0.T, "Atcov_%d_%d"%(i, step))
      u.dump(s0, "As_%d_%d"%(i, step))

    for i in range(1, n+1):
      svd = vars_svd_B2[i]
      s0, u0, v0 = sess.run([svd.s, svd.u, svd.v])
      u.dump(s0, "B2_%d_%d"%(i, step))
      B0 = B[i].eval()
      Bt0 = v0.T @ B0
      u.dump(B0 @ B0.T, "Bcov_%d_%d"%(i, step))
      u.dump(Bt0 @ Bt0.T, "Btcov_%d_%d"%(i, step))
      u.dump(s0, "Bs_%d_%d"%(i, step))      
    
  def advance_batch():
    sess.run(sampled_labels.initializer)  # new labels for next call

  def update_covariances():
    ops_A = [cov_A[i].initializer for i in range(1, n+1)]
    ops_B2 = [cov_B2[i].initializer for i in range(1, n+1)]
    sess.run(ops_A+ops_B2)

  def update_svds():
    vars_svd_A[2].update()
    vars_svd_B2[2].update()
    vars_svd_B2[1].update()

  def init_svds():
    """Initialize our SVD to identity matrices."""
    ops = []
    for i in range(1, n+1):
      ops.extend(vars_svd_A[i].init_ops)
      ops.extend(vars_svd_B2[i].init_ops)
    sess = tf.get_default_session()
    sess.run(ops)
      
  init_op = tf.global_variables_initializer()

  from tensorflow.core.protobuf import rewriter_config_pb2
  
  rewrite_options = rewriter_config_pb2.RewriterConfig(
    disable_model_pruning=True,
    constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
    memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  graph_options=tf.GraphOptions(optimizer_options=optimizer_options,
                                rewrite_options=rewrite_options)
  config = tf.ConfigProto(graph_options=graph_options)

  sess = tf.InteractiveSession(config=config)
  sess.run(Wf.initializer, feed_dict=init_dict)
  sess.run(X.initializer, feed_dict=init_dict)
  advance_batch()
  update_covariances()
  init_svds()
  sess.run(init_op, feed_dict=init_dict)  # initialize everything else
  
  print("Running training.")
  u.reset_time()

  step_lengths = []     # keep track of learning rates
  losses = []
  
  # adaptive line search parameters
  alpha=0.3   # acceptable fraction of predicted decrease
  beta=0.8    # how much to shrink when violation
  growth_rate=1.05  # how much to grow when too conservative
    
  def update_cov_A(i):
    sess.run(cov_A[i].initializer)
  def update_cov_B2(i):
    sess.run(cov_B2[i].initializer)

  # only update whitening matrix of input activations in the beginning
  vars_svd_A[1].update()

  for step in range(40): 
    update_covariances()
    update_svds()

    sess.run(grad.initializer)
    sess.run(pre_grad.initializer)
    
    lr0, loss0 = sess.run([lr, loss])
    update_params_op.run()
    advance_batch()

    losses.append(loss0)
    step_lengths.append(lr0)

    print("Step %d loss %.2f"%(step, loss0))
    u.record_time()

  assert losses[-1]<0.59
  assert losses[-1]>0.57
  assert 20e-3<min(u.global_time_list)<50e-3, "Time should be 40ms on 1080"
  u.summarize_time()
  print("Test passed")


if __name__=='__main__':
  main()

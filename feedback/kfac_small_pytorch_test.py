# Times: min: 440.60, median: 452.39, mean: 453.87
# macbook timing: 1046.48
# switching randn: 1023.13
# caching first A: 995.28

# workstation
# Times: min: 470.33
# switching to native inverse
# Times: min: 486.14, median: 488.69, mean: 492.51
#  GPU inverse: 55.28
# after switching to mkl inverse: 42.24
# after preallocating noise tensor: 36.85,
# biggest bottleneck is inverses, about 10ms per inverse

import util as u
u.check_mkl()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import scipy
import sys

from torch.autograd.function import Function

# for line profiling
try:
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.


import common_gd
args = common_gd.args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
  print('using cuda')
        
# model options
dtype = np.float32
torch_dtype = 'torch.FloatTensor'
if args.cuda:
  torch_dtype = 'torch.cuda.FloatTensor'

lambda_=3e-3
lr = 0.2
dsize = 10000
nonlin = torch.sigmoid

# debugging options
INVERSE_METHOD = 'numpy'  # cpu, numpy, gpu
DO_PRINT = False

def _get_output(ctx, arg, inplace=False):
  if inplace:
    ctx.mark_dirty(arg)
    return arg
  else:
    return arg.new().resize_as_(arg)

forward_list = []
backward_list = []

class Addmm(Function):
        
  @staticmethod
  @profile
  def forward(ctx, add_matrix, matrix1, matrix2, beta=1, alpha=1, inplace=False):
    ctx.save_for_backward(matrix1, matrix2)
    output = _get_output(ctx, add_matrix, inplace=inplace)
    forward_list.append(matrix2)
    return torch.addmm(beta, add_matrix, alpha,
                       matrix1, matrix2, out=output)

  @staticmethod
  @profile
  def backward(ctx, grad_output):
    matrix1, matrix2 = ctx.saved_variables
    grad_matrix1 = grad_matrix2 = None

    if ctx.needs_input_grad[1]:
      grad_matrix1 = torch.mm(grad_output, matrix2.t())

    if ctx.needs_input_grad[2]:
      grad_matrix2 = torch.mm(matrix1.t(), grad_output)

    if DO_PRINT:
      print("backward got")
      print("grad_output", grad_output)
      print("matrix1", matrix1)
      print('matrix2', matrix2)
      print('grad_matrix1', grad_matrix1)

    # insert dsize correction to put activations/backprops on same scale      
    backward_list.append(grad_output*dsize)
    return None, grad_matrix1, grad_matrix2, None, None, None

@profile
def my_matmul(mat1, mat2):
  output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
  return Addmm.apply(output, mat1, mat2, 0, 1, True)

@profile
def regularized_inverse(mat):
  assert mat.shape[0] == mat.shape[1]
  ii = torch.eye(mat.shape[0])
  if args.cuda:
    ii = ii.cuda()
  regmat = mat + lambda_*ii

  if INVERSE_METHOD == 'numpy':
    result = torch.from_numpy(scipy.linalg.inv(regmat.cpu().numpy()))
    if args.cuda:
      result = result.cuda()
  elif INVERSE_METHOD == 'gpu':
    result = torch.inverse(regmat)
  else:
    assert False, 'unknown INVERSE_METHOD ' + str(INVERSE_METHOD)
  return result


def t(mat): return torch.transpose(mat, 0, 1)

def copy_list(l):
  new_list = []
  for item in l:
  #    new_list.append(np.copy(l.numpy()))
    new_list.append(item.clone())
  return new_list

@profile
def main():
  global forward_list, backward_list, DO_PRINT
  
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      W0 = u.ng_init(196, 784)
      W1 = u.ng_init(784, 196)  # fix non-contiguous input
      self.W0 = nn.Parameter(torch.from_numpy(W0))
      self.W1 = nn.Parameter(torch.from_numpy(W1))

    def forward(self, input):
      x = input.view(784, -1)
      x = nonlin(my_matmul(self.W0, x))
      x = nonlin(my_matmul(self.W1, x))
      return x.view_as(input)

  model = Net()
  if args.cuda:
    model.cuda()

  data0 = u.get_mnist_images()
  data0 = data0[:, :dsize].astype(dtype)
  data = Variable(torch.from_numpy(np.copy(data0)).contiguous())
  if args.cuda:
    data = data.cuda()

  model.train()
  optimizer = optim.SGD(model.parameters(), lr=lr)
  losses = []
  n = 2
  
  covA = [None]*n
  covA_inv = [None]*n

  noise = torch.Tensor(*data.data.shape).type(torch_dtype)
  for step in range(10):
    optimizer.zero_grad()
    forward_list = []
    backward_list = []
    output = model(data)
    err = output-data
    loss = torch.sum(err*err)/2/dsize
    loss.backward(retain_graph=True)
    loss0 = loss.data[0]

    A = forward_list[:]
    B = backward_list[::-1]
    assert len(B) == n

    
    forward_list = []
    backward_list = []
    
    #    noise = torch.randn(*data.data.shape)
    #    torch.randn(2,2).type('torch.cuda.FloatTensor')
    noise.normal_()
    synthetic_data = Variable(output.data+noise)
    # todo, not needed?
    if args.cuda:
      synthetic_data = synthetic_data.cuda()
      
    err2 = output - synthetic_data
    loss2 = torch.sum(err2*err2)/2/dsize
    optimizer.zero_grad()
    backward_list = []
    loss2.backward()
    B2 = backward_list[::-1]
    assert len(B2) == n


    # compute whitened gradient
    pre_dW = []
    for i in range(n):
      # only compute first activation once
      if i > 0:
        covA[i] = A[i] @ t(A[i])/dsize
        covA_inv[i] = regularized_inverse(covA[i])
      else:
        if covA[i] is None:
          covA[i] = A[i] @ t(A[i])/dsize
          covA_inv[i] = regularized_inverse(covA[i])
          
      #      else:
      covB2 = B2[i]@t(B2[i])/dsize
      covB = B[i]@t(B[i])/dsize
      whitened_A = covA_inv[i]@A[i]
      whitened_B = regularized_inverse(covB2.data)@B[i].data
      pre_dW.append(whitened_B @ t(whitened_A)/dsize)

    params = list(model.parameters())
    assert len(params) == len(pre_dW)
    for i in range(len(params)):
      params[i].data-=lr*pre_dW[i]
    
    print("Step %3d loss %10.9f"%(step, loss0))
    u.record_time()

  loss0 = loss.data.cpu().numpy()#[0]
  target = 2.360062122
  
  if 'Apple' in sys.version:
    target = 2.360126972
    target = 2.335654736  # after changing randn
  if args.cuda:
    target = 2.337174654
    target = 2.337215662  # switching to GPU inverse
    
  u.summarize_time()
  assert abs(loss0-target)<1e-9, abs(loss0-target)
    

if __name__=='__main__':
  main()

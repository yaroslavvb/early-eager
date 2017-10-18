import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import scipy

from torch.autograd.function import Function

import common_gd
args = common_gd.args
args.cuda = not args.no_cuda and torch.cuda.is_available()

dtype = np.float32
nonlin = torch.sigmoid
nonlin = F.relu
nonlin = lambda x: x

def _get_output(ctx, arg, inplace=False):
  if inplace:
    ctx.mark_dirty(arg)
    return arg
  else:
    return arg.new().resize_as_(arg)

forward_list = []
backward_list = []
lambda_=3e-3
lr = 0.2
dsize = 2 # 1000

class Addmm(Function):
        
  @staticmethod
  def forward(ctx, add_matrix, matrix1, matrix2, beta=1, alpha=1, inplace=False):
    ctx.save_for_backward(matrix1, matrix2)
    output = _get_output(ctx, add_matrix, inplace=inplace)
    forward_list.append(matrix2)
    return torch.addmm(beta, add_matrix, alpha,
                       matrix1, matrix2, out=output)

  @staticmethod
  def backward(ctx, grad_output):
    matrix1, matrix2 = ctx.saved_variables
    grad_matrix1 = grad_matrix2 = None

    if ctx.needs_input_grad[1]:
      grad_matrix1 = torch.mm(grad_output, matrix2.t())

    if ctx.needs_input_grad[2]:
      grad_matrix2 = torch.mm(matrix1.t(), grad_output)

##    print("backward got")
#    print("grad_output", grad_output)
#    print("matrix1", matrix1)
#    print('matrix2', matrix2)
#    print('grad_matrix1', grad_matrix1)

    # insert dsize correction to put activations/backprops on same scale
    backward_list.append(grad_output*dsize)
    return None, grad_matrix1, grad_matrix2, None, None, None


def my_matmul(mat1, mat2):
  output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
  return Addmm.apply(output, mat1, mat2, 0, 1, True)

def inverse(mat):
  assert mat.shape[0] == mat.shape[1]
  regmat = mat + lambda_*torch.eye(mat.shape[0])
  return torch.from_numpy(scipy.linalg.inv(regmat.numpy()))


def t(mat): return torch.transpose(mat, 0, 1)

def main():
  global forward_list, backward_list
  
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed(args.seed)
  # images = torch.Tensor(u.get_mnist_images())
  # images = images[:dsize]
  # if args.cuda:
  #   images = images.cuda()
  data0 = np.array([[0., 1], [2, 3]]).astype(dtype)
  data = Variable(torch.from_numpy(data0))

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      W0 = (np.array([[0., 1], [2, 3]])/10).astype(dtype)
      #      W1 = (np.array([[4., 5], [6, 7]])/10).astype(dtype)
      self.W0 = nn.Parameter(torch.from_numpy(W0))
      #      self.W1 = nn.Parameter(torch.from_numpy(W1))

    def forward(self, input):
      x = input.view(-1, 2)
      x = nonlin(my_matmul(self.W0, x))
      #      x = nonlin(my_matmul(self.W1, x))
      return x.view_as(input)

  # initialize model and weights
  model = Net()
#  params1, params2 = list(model.parameters())
  if args.cuda:
    model.cuda()
  
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=lr)
  for step in range(2):
    optimizer.zero_grad()
    forward_list = []
    output = model(data)
    err = output-data
    #    loss = F.mse_loss(output, data)*dsize # equiv to L2 * dsize/2
    loss = torch.sum(err*err)/2/dsize
    #    sampled_labels = Variable(torch.Tensor(np.random.normal(size=4).reshape((2,2))))
#    #    print('sampled_labels', sampled_labels)
    backward_list = []
    #    loss2 = F.mse_loss(output, output+sampled_labels)*dsize/2 # equiv to L2 * dsize/2
#    #    print('loss2', loss2)
    #    err = output - data
    #    loss = torch.sum(err*err)/(2*dsize)
    loss0 = loss.data[0]
    loss.backward()
#    print("forwards")
#    print(forward_list)
#    print('backwards')
#    print(backward_list)
#    print('grads')
#    print(model.W0.grad.data)

    # compute whitened gradient
    pre_dW = []
    for (A,B) in zip(forward_list, reversed(backward_list)):
      covA = A @ t(A)
      #      covB = B@torch.transpose(B)
      covA_inv = inverse(covA)
      whitened_A = A #regularized_inverse(covA)@A
      whitened_B = B.data
      pre_dW.append(whitened_B @ t(whitened_A)/dsize)
      print('A', A)
      print('B', B)
      print('dW', pre_dW)
      

    params = list(model.parameters())
    assert len(params) == len(pre_dW)
#    for i in range(len(params)):
#      params[i].data-=lr*pre_dW[i]
    optimizer.step()
    
    print("Step %3d loss %10.9f"%(step, loss0))
    u.record_time()

  target = 0.275399953
  assert abs(loss0-target)<1e-5, abs(loss0-target)
  u.summarize_time()
    

if __name__=='__main__':
  main()

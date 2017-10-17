import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

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

class Addmm(Function):
        
  @staticmethod
  def forward(ctx, add_matrix, matrix1, matrix2, beta=1, alpha=1, inplace=False):
    ctx.save_for_backward(matrix1, matrix2)
    output = _get_output(ctx, add_matrix, inplace=inplace)
    forward_list.append(output)
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

    backward_list.append(grad_matrix1)
    return None, grad_matrix1, grad_matrix2, None, None, None


def my_matmul(mat1, mat2):
  output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
  return Addmm.apply(output, mat1, mat2, 0, 1, True)


def main():
  global forward_list, backward_list
  
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  lambda_=3e-3
  lr = 0.2
  dsize = 2 # 1000
  # images = torch.Tensor(u.get_mnist_images())
  # images = images[:dsize]
  # if args.cuda:
  #   images = images.cuda()
  data0 = np.array([[0., 1], [2, 3]]).astype(dtype)
  data = Variable(torch.Tensor(data0))

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.W0 = nn.Parameter(torch.Tensor([[0., 1], [2, 3]])/10)
      self.W1 = nn.Parameter(torch.Tensor([[4., 5], [6, 7]])/10)

    def forward(self, input):
      x = input.view(-1, 2)
      x = nonlin(my_matmul(self.W0, x))
      x = nonlin(my_matmul(self.W1, x))
      return x.view_as(input)

  # initialize model and weights
  model = Net()
  params1, params2 = list(model.parameters())
  if args.cuda:
    model.cuda()
  
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=lr)
  for step in range(10):
    optimizer.zero_grad()
    forward_list = []
    output = model(data)
    loss = F.mse_loss(output, data)*dsize/2 # equiv to L2 * dsize/2
    #    sampled_labels = Variable(torch.Tensor(np.random.normal(size=4).reshape((2,2))))
    #    print('sampled_labels', sampled_labels)
    backward_list = []
    #    loss2 = F.mse_loss(output, output+sampled_labels)*dsize/2 # equiv to L2 * dsize/2
    #    print('loss2', loss2)
    #    err = output - data
    #    loss = torch.sum(err*err)/(2*dsize)
    loss0 = loss.data[0]
    loss.backward()
    #    print(forward_list)
    #    print(backward_list)
    optimizer.step()
    
    print("Step %3d loss %10.9f"%(step, loss0))
    u.record_time()

  u.summarize_time()
    

if __name__=='__main__':
  main()

import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from torch.autograd.function import Function

dtype = np.float64

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

    #    backward_list.append(grad_matrix1)
    print('grad_output', grad_output)
    print('grad_matrix1', grad_matrix1)
    return None, grad_matrix1, grad_matrix2, None, None, None


def my_matmul(mat1, mat2):
  output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
  return Addmm.apply(output, mat1, mat2, 0, 1, True)


def main():
  global forward_list, backward_list
  
  torch.manual_seed(1)
  np.random.seed(1)

  lr = 0.2
  data0 = np.array([[0., 1], [2, 3]]).astype(dtype)
  data = Variable(torch.from_numpy(data0))

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      W0_0 = np.array([[0., 1], [2, 3]], dtype=dtype)/10
      self.W0 = nn.Parameter(torch.from_numpy(W0_0))

    def forward(self, input):
      x = input.view(-1, 2)
      x = my_matmul(self.W0, x)
      return x.view_as(input)

  # initialize model and weights
  model = Net()
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=lr)
  for step in range(2):
    optimizer.zero_grad()
    forward_list = []
    output = model(data)
    err = output-data
    loss = torch.sum(err*err)
    loss0 = loss.data[0]
    loss.backward()
    print('loss', loss0)
    print('W0', model.W0)
    print('X', data.data)
    print('err', err.data)
    print('grad', model.W0.grad)
    desired_result = np.array([[ -1.4,  -3.4], [ -3.8, -17. ]],
                              dtype=dtype)
    #
    #np.testing.assert_allclose(model.W0.grad.data.numpy(), desired_result)
    optimizer.step()
    
    print("Step %3d loss %10.9f"%(step, loss0))

  assert loss0-116.301605225<1e-9


if __name__=='__main__':
  main()

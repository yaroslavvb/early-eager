import torch
from torch.autograd import Variable

from torch.autograd.function import Function

def _get_output(ctx, arg, inplace=False):
  if inplace:
    ctx.mark_dirty(arg)
    return arg
  else:
    return arg.new().resize_as_(arg)
    
class Addmm(Function):
        
  @staticmethod
  def forward(ctx, add_matrix, matrix1, matrix2, beta=1, alpha=1, inplace=False):
    ctx.save_for_backward(matrix1, matrix2)
    output = _get_output(ctx, add_matrix, inplace=inplace)
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
    return None, grad_matrix1, grad_matrix2, None, None, None


def my_matmul(mat1, mat2):
  output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
  return Addmm.apply(output, mat1, mat2, 0, 1, True)


def main():
  n = 2
  lr = 0.25
  x = Variable(torch.ones((n, n)), requires_grad=True)
  y = Variable(torch.Tensor([[1, 2], [3, 4]]))

  for step in range(5):
    loss = torch.sum(my_matmul(x, y))
    loss.backward()
    loss0 = loss.data.numpy()[0]
    print("loss =", loss0)
    x.data-=lr*x.grad.data
    x.grad.data.zero_()
  assert loss0 == -96

if __name__=='__main__':
  main()

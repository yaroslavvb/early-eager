import torch
from torch.autograd import Variable

from torch.autograd.function import Function
    
def main():
  n = 2
  lr = 0.25
  x = Variable(torch.ones((n, n)), requires_grad=True)
  y = Variable(torch.Tensor([[1, 2], [3, 4]]))

  for step in range(5):
    loss = torch.sum(torch.mm(x, y))
    loss0 = loss.data.numpy()[0]
    print("loss =", loss0)
    manual_grad = torch.ones((2, 2))@torch.transpose(y.data, 0, 1)
    x.data-=lr*manual_grad
  assert loss0 == -96

if __name__=='__main__':
  main()

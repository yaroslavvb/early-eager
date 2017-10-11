import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import common_gd
args = common_gd.args
args.cuda = not args.no_cuda and torch.cuda.is_available()

step = 0
def main():
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  
  images = torch.Tensor(u.get_mnist_images().T)
  images = images[:args.batch_size]
  if args.cuda:
    images = images.cuda()
  data = Variable(images)

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.encoder = nn.Parameter(torch.rand(args.visible_size,
                                             args.hidden_size))

    def forward(self, input):
      x = input.view(-1, args.visible_size)
      x = torch.sigmoid(torch.mm(x, self.encoder))
      x = torch.sigmoid(torch.mm(x, torch.transpose(self.encoder, 0, 1)))
      return x.view_as(input)

  # initialize model and weights
  model = Net()
  model.encoder.data = torch.Tensor(u.ng_init(args.visible_size,
                                              args.hidden_size))
  if args.cuda:
    model.cuda()
  
  model.train()
  if args.gd:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
  else:
    optimizer = optim.LBFGS(model.parameters(), max_iter=args.iters, lr=args.lr)

  def closure():
    global step
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, data)
    loss0 = loss.data[0]
    print("step %3d loss %6.5f"%(step, loss0))
    step+=1
    loss.backward()
    u.record_time()
    return loss
  
  if args.gd:
    iters = args.iters
  else:
    iters = 1
  for i in range(iters):
    optimizer.step(closure)
    
  u.summarize_time()
    

if __name__=='__main__':
  main()

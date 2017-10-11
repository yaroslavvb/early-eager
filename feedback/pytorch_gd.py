import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import common_gd
args = common_gd.args

def main():
  torch.manual_seed(0)
  np.random.seed(0)
  
  images = u.get_mnist_images().T 
  images = images[:args.batch_size]
  data = Variable(torch.Tensor(images))

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.encoder = nn.Linear(args.visible_size, args.hidden_size, bias=False)
      self.decoder = nn.Linear(args.hidden_size, args.visible_size, bias=False)

    def forward(self, input):
      x = input.view(-1, args.visible_size)
      x = self.encoder(x)
      x = F.sigmoid(x)
      x = self.decoder(x)
      x = F.sigmoid(x)
      return x.view_as(input)

  # initialize model and weights
  model = Net()
  params1, params2 = list(model.parameters())
  params1.data = torch.Tensor(u.ng_init(args.visible_size, args.hidden_size).T)
  params2.data = torch.Tensor(u.ng_init(args.hidden_size, args.visible_size).T)
  
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=args.lr)
  for step in range(args.iters):
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, data)
    loss0 = loss.data[0]
    loss.backward()
    optimizer.step()
    
    print("Step %3d loss %6.5f"%(step, loss0))
    u.record_time()

  u.summarize_time()
    

if __name__=='__main__':
  main()

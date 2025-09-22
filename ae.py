import torch
import torch.nn as nn
import torch.nn.functional as F

#HYPERPARAMS:
lr = 1e-3
batch_size = 512
weight_decay = 1e-5
epochs = 200
r_lambda = 0.01
#Can't read the sentence on pg.19

#######################################
#For now, hard code layers as described in page 19, section G.1 of https://arxiv.org/pdf/2502.13257
#######################################
class AE(nn.Module):
  def __init__(self, inp_size):
    super(AE, self).__init__()
    self.inp_scale_1 = nn.Linear(inp_size, 800)
    self.inp_scale_2 = nn.Linear(800, inp_size)
    self.f = nn.Sequential(
      nn.Linear(800, 400),
      nn.ELU(),
      nn.Linear(400, 100),
      nn.ELU(),
      nn.Linear(100, 2)
    )
    self.g = nn.Sequential(
      nn.Linear(2, 100),
      nn.ELU(),
      nn.Linear(100, 400),
      nn.ELU(),
      nn.Linear(100, 2),
      nn.Softmax()
    )
  def forward(self, p_x):
    z_0 = self.inp_scale_1(p_x)
    z = self.f(z_0)
    p_x_0 = self.g(z)
    p_x_r = self.inp_scale_2(p_x_0)
    return p_x_r, z


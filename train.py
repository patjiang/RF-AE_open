import torch
from torch.utils.data import DataLoader

from ae import *
from loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temp_model = AE().to(device)
optim = torch.optim.Adam(temp_model.parameter(), lr = 1e-3)
epochs = 200
dataset = torch.randn(1000, 800).to(device)

for epoch in range(epochs):
  for data in dataset:
    recon, z = AE(data)
    #ergh gotta go to next class

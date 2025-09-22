import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import statistics

from ae import *
from loss import *

def train_step(AE, data):
  recon, z = AE(data)
  #@TODO: Replace with RF-PHATE
  dummy_z = torch.randn(1, 2).to(device)
  loss = compound_loss((data, recon), (z, dummy_z))
  return loss

def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  temp_model = AE().to(device)
  optim = torch.optim.Adam(temp_model.parameter(), lr = 1e-3)
  epochs = 200
  #TODO: Replace with datast object that uses RF-GAP from raw data to prob. measures
  dataset = torch.randn(1000, 800).to(device)
  loss = []
  for epoch in tqdm(range(epochs)):
    epclosses = []
    for data in dataset:
      optim.zero_grad()
      loss = train_step(AE, data)
      loss.backward()
      optim.step()
      epclosses.append(loss.detach().cpu().item())
    mu, var = statistics.mean(epclosses), statistics.stdev(epclosses)
    printf('epoch {epoch}, mean = {mu:.3f}, var = {var:.3f}')
    loss.append(mu)
  return loss

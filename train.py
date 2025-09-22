import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import statistics

from .ae import *
from .loss import *

def train_step(model, data):
  recon, z = model(data)
  #@TODO: Replace with RF-PHATE
  dummy_z = torch.randn(1, 2).to(device)
  loss = compound_loss((data, recon), (z, dummy_z))
  return loss
  
#TODO: Replace with datast object that uses RF-GAP from raw data to prob. measures
def train(inp_size=800, epochs = 200, dataset = torch.randn(1000, 800), device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
  temp_model = AE(inp_size).to(device)
  optim = torch.optim.Adam(temp_model.parameters(), lr = 1e-3)
  loss = []
  for epoch in tqdm(range(epochs)):
    epclosses = []
    for data in dataset:
      optim.zero_grad()
      loss = train_step(temp_model, data.to(device))
      loss.backward()
      optim.step()
      epclosses.append(loss.detach().cpu().item())
    mu, var = statistics.mean(epclosses), statistics.stdev(epclosses)
    printf('epoch {epoch}, mean = {mu:.3f}, var = {var:.3f}')
    loss.append(mu)
  return loss

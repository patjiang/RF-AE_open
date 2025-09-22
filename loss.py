import torch
import torch.nn.functional as F

#######################################
# Define initial loss functions
#######################################

def reconstruction_loss(p, p_hat):
  m = (p.add(p_hat)).div(2)
  kl_pm = F.kl_div(p.log(), m)
  kl_p_hat_m = F.kl_div(p_hat.log(), m)
  return (kl_pm.add(kl_p_hat_m)).div(2)


def geom_reg_loss(z, z_hat):
  return ((z.sub(z_hat))**2).sum(dim=list(range(1, z.dim())))


def compound_loss(ps, zs, r_lambda=0.01):
  p, ph = ps
  z, zh = zs
  recon = reconstruction_loss(p, ph).mult(r_lambda)
  geoml = geom_reg_loss(z, zh).mult(1-r_lambda)
  return (recon.add(geoml)).mean()

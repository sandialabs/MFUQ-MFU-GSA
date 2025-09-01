# %%
import pytest
import os, sys
import numpy as np

sys.path.insert(0,'../')
from generalizedADE import generalizedADE, FRADE

# %%
def test_FRADE_solve():
  """
  Checking can reproduce analytical solution to ADE 
  with cosine initial condition. 

  IC = cos( 2 pi (Nk-1) / Lx  x )
  c(x,t) = exp( - nu (2 pi (Nk-1)/ Lx)^2 t )cos( 2 pi (Nk - 1) / Lx (x - ut)) 
  """
  model = FRADE()
  params = np.array( [None, None, None, 0.003, 2.0])

  #k = model.Nk-2 # Numerically this started degrading because things had decayed so much
  k = 10
  model.t = 0.01
  model.IC = np.cos( 2 * np.pi * k / model.Lx * model.x ) 
  model.IC_coeffs = np.fft.rfft( model.IC )
  c_computed = model.f_field(params)

  nu = model.nu_p + params[3]
  c_analytical = np.exp( - nu * (2 * np.pi * k / model.Lx )**2. * model.t ) * np.cos(
      2 * np.pi * k / model.Lx * (model.x - model.u * model.t))

  rel_err = np.linalg.norm(c_analytical-c_computed)/np.linalg.norm(c_analytical)
  print(f"ADE test relative error: {rel_err:.2e}")
  assert rel_err < 1e-14

# %%
def test_generalizedADE_solve():
  """
  Checking can reproduce analytical solution to ADE 
  with cosine initial condition. 

  IC = cos( 2 pi (Nk-1) / Lx  x )
  c(x,t) = exp( - nu (2 pi (Nk-1)/ Lx)^2 t )cos( 2 pi (Nk - 1) / Lx (x - ut)) 
  """
  model = generalizedADE()
  eigenvalues = (0.003*model.ikx**2.0)[1:-1]
  params = np.hstack( ([None, None, None], eigenvalues))

  #k = model.Nk-2 # Numerically this started degrading because things had decayed so much
  k = 10
  model.t = 0.01
  model.IC = np.cos( 2 * np.pi * k / model.Lx * model.x ) 
  model.IC_coeffs = np.fft.rfft( model.IC )
  c_computed = model.f_field(params)

  nu = model.nu_p + .003
  c_analytical = np.exp( - nu * (2 * np.pi * k / model.Lx )**2. * model.t ) * np.cos(
      2 * np.pi * k / model.Lx * (model.x - model.u * model.t))

  rel_err = np.linalg.norm(c_analytical-c_computed)/np.linalg.norm(c_analytical)
  print(f"ADE test relative error: {rel_err:.2e}")
  assert rel_err < 1e-14

# %%
def test_FRADE_and_generalizedADE_identical():
  """
  Checking get the same solution from FRADE and generalizedADE if set the parameters the same.
  """

  FRADE_model = FRADE()
  gADE_model = generalizedADE()

  FRADE_params = np.array( [None, None, None, 0.05, 1.5]) 
  FRADE_y = FRADE_model.f_field(FRADE_params)

  eigenvalues = (.05*gADE_model.ikx**1.5)[1:-1]
  gADE_params = np.hstack( ([None, None, None], eigenvalues))
  gADE_y = gADE_model.f_field(gADE_params)

  rel_err = np.linalg.norm(FRADE_y - gADE_y)/np.linalg.norm(FRADE_y)
  print(f"FRADE vs generalizedADE relative error: {rel_err:.2e}")
  assert rel_err == 0
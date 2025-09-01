# %%
import pytest
import os, sys
import numpy as np

sys.path.insert(0,'../')
from MFU_representations import FRADESobols, generalizedADESobolsfromFRADE, generalizedADEMVNormalSobols, generalizedADEDataConsistentSobols

#%%
def test_sampling():
  """
  Checking can get the same output samples from FRADESobols and generalizedADESobolsfromFRADE

  These should be identical because we are using the FRADE assumption to compute the eigenvalues 
  for the generalized ADE.
  """

  N = 10

  np.random.seed(20221108)
  FRADE_model = FRADESobols()
  FRADE_X = FRADE_model.get_input_samples(N)
  FRADE_y = np.zeros(N)
  for i in range(N):
    FRADE_y[i] = FRADE_model.f(FRADE_X[i,:])
  
  np.random.seed(20221108)
  generalizedADE_model = generalizedADESobolsfromFRADE()
  generalizedADE_X = generalizedADE_model.get_input_samples(N)
  generalizedADE_y = np.zeros(N)
  for i in range(N):
    generalizedADE_y[i] = generalizedADE_model.f(generalizedADE_X[i,:])

  err = np.linalg.norm( generalizedADE_y - FRADE_y)
  print(f"Err between FRADE and generalizedADE samples: {err:.2e}")
  assert err == 0

# %%
# Checking all the SobolIndices child classes run.
def test_FRADESobols_runs():
    m = FRADESobols()
    m.compute_sobol_indices(10)
    assert True

def test_generalizedADESobolsfromFRADE_runs():
    m = generalizedADESobolsfromFRADE()
    m.compute_sobol_indices(10)
    assert True

def test_generalizedADEMVNormalSobols_runs():
    m = generalizedADEMVNormalSobols()
    m.compute_sobol_indices(10)
    assert True

def test_generalizedADEDataConsistentSobols_runs():
    m = generalizedADEDataConsistentSobols()
    m.compute_sobol_indices(10)
    assert True

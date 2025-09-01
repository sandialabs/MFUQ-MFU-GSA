# %%
import pytest
import os, sys
import numpy as np
import itertools

# %%
sys.path.insert(0,'../')
from SobolIndices import *

# %%
class pairwiseMultiplicativeModel(SobolIndices):
  """
  Implementing analytical test problem from [3].
  y = a * x1 * x2 + b * x3 * x4 + c * x5 * x6

  xi ~ N(0,1) for i=1-6; corr(x3,x4) = rho34, corr(5,6)=rho56

  S1 = 0, S2 = 0 (if either is fixed, expectation over the other 
  evaluates to 0; e.g. E_x1[y|x2] = a*x1*E[x2] = a*x1*0 = 0)

  Grouped indices and higher-order interaction terms follow:

  V(y) = V = a^2 + b^2(1+rho34^2) + c^2(1 + rho56^2)

  S_12 = a^2 / V
  S_{3,4} = b^2 ( 1 + rho34^2 ) / V
  S_{5,6} = c^2 ( 1 + rho56^2 ) / V
  """
  def __init__(self):
      parameter_groups = [['x1'],['x2'],['x3','x4'],['x5','x6']]
      super().__init__(parameter_groups)

      self.a = 1.1
      self.b = 2.0
      self.c = 3.0

      self.rho34 = 0.8
      self.rho56 = 0.3

      self.V = self.a**2 + self.b**2 * (1+self.rho34**2) + self.c**2 * (1+self.rho56**2)
      self.S12 = self.a**2 / self.V
      self.S3c4 = self.b**2 * (1+self.rho34**2) / self.V
      self.S5c6 = self.c**2 * (1+self.rho56**2) / self.V

      self.S = [ 0, 0, self.S3c4, self.S5c6]


  # Virtual function to evaluate the model f:R^p->R
  def f(self,x):
      y = (self.a * x[0] * x[1]) + (self.b * x[2] * x[3]) + (self.c * x[4] * x[5])
      return y

  # Virtual function to sample the input space
  def get_input_samples(self,N):
      X = np.zeros((N,6))
      X[:,:2] = np.random.standard_normal(size=(N,2))
      X[:,2:4] = np.random.multivariate_normal([0,0], [[1,self.rho34],[self.rho34,1]], size=N )
      X[:,4:6] = np.random.multivariate_normal([0,0], [[1,self.rho56],[self.rho56,1]], size=N )
      return X

# %%
class SobolG(SobolIndices):
  """
  Implementing the Sobol' G function discussed in [1,2]

  G = prod_i=1^p g_i, 

  g_i = ( |4 x_i -2| + a_i ) / (1 + a_i )
  
  Let mu'_2,i is the 2nd raw moment of g_i: 
    mu'_2,i = 1 + 1/3*(a_i+1)^(-2) 

  Then 
    Vu = ( prod_(i in u) mu'_2,i ) - 1
    V = ( prod_(i=1)^p mu'_2,i ) - 1
  and
  Su = Vu / V
  Tu = 1 - Vuc / V, where uc = i not in u

  a >= 9 => not important
  a ~ 0 => very important
  multiple a ~ 0 => interactions important
  """
  def __init__(self, 
      a=np.array([0,0.1,0.2,0.3,0.4,0.8,1,2,3,4]), 
      parameter_groups = [['x1','x2','x3'],['x4','x5'],['x6','x7', 'x8', 'x9', 'x10']]
      ):
    super().__init__(parameter_groups)
    
    self.a = a
    # The 2nd raw moment of g (lowercase intentional) function computed element-wise
    self.muprime2 = 1. + (self.a + 1.)**(-2.)/3.
    self.V = np.prod( self.muprime2 )-1

    self.get_analytical_indices()

  def get_analytical_main_index(self, group_inds):
    # (prod_(k in u) muprime2 - 1 ) / V
    return ( np.prod(self.muprime2[group_inds]) - 1 ) / self.V

  def get_analytical_total_index(self,group_inds):
    # The set of indices not in group_inds
    uc_inds = [ i for i in np.arange(self.n_params) if not i in group_inds ]
    # Tu = 1 - Suc
    return 1 - self.get_analytical_main_index(uc_inds)

  def get_analytical_indices( self ):
    self.Sa = np.zeros((self.n_groups,))
    self.Ta = np.zeros((self.n_groups,))
    for i, g_inds in enumerate(self.g_inds):
      self.Sa[i] = self.get_analytical_main_index(g_inds)
      self.Ta[i] = self.get_analytical_total_index(g_inds)

  def get_analytical_interaction_effect( self, ind1, ind2 ):
    
    S1 = self.get_analytical_main_index(ind1)
    S2 = self.get_analytical_main_index(ind2)
    Su = self.get_analytical_main_index([ind1,ind2])
    return Su - S1 - S2

  def g_function(self, x):
    # Applies the g function 
    #    g_i = ( |4 x_i -2| + a_i ) / (1 + a_i )
    # elementwise to the vector x
    return ( np.abs( 4.*x - 2. ) + self.a ) / (1. + self.a )

  def f(self,x):
    # Applies the G function (just product of each g function)
    return np.prod(self.g_function(x))

  def get_input_samples(self, N):
    return np.random.uniform(0,1,size=(N,self.n_params))

# %%
class SobolG_2Outputs(SobolG):
  def __init__(self):
    super().__init__()

    self.n_outputs = 2
    self.output_names = ['G', '2G']
  
  def f(self, x):
    gf = super().f(x)
    return np.array([gf, 2*gf])

# %%
def test_analytical_sobols():
  """
  Check can reproduce grouped Sobol' indices for analytical case, from [1], 
  section 4 and [2].
  """
  m = SobolG( a=np.array([ 0,0,3,3,3,3,3,3 ]), 
    parameter_groups=[[f"{i}"] for i in range(8)]
    )
  Strue = np.array([0.329, 0.329, 0.021, 0.021, 0.021, 0.021, 0.021, 0.021])
  err = np.linalg.norm( Strue - np.round(m.Sa,3))
  print(f"\nTrue main effects:       {Strue}")
  print(f"Analytical main effects: {np.round(m.Sa,3)}")
  assert err == 0.0

  ## Higher order terms for computing total indices. 
  ## Note that there are higher order indices that aren't 
  ## enumerated here, but they do not exceed 0.00014, so we expect
  ## error to be < 1e-3 between analytical and what we compute here.
  #S12 = 0.110
  #S13 = S14 = S15 = S16 = S17 = S18 = 0.007
  #S23 = S24 = S25 = S26 = S27 = S28 = 0.007
  #S34 = S36 = S37 = S38 = S45 = S46 = S47 = S48 = 0.0004
  #S123 = S124 = S125 = S126 = S127 = S128 = 0.002

  #T1 = Strue[0] + S12 + S13 + S14 + S15 + S16 + S17 + S18 + S123 + S124 + S125 + S126 + S127 + S128
  #print(np.round(m.T[0],3))
  #print(m.T)

# %%
def test_grouped_sobols():
  """
  Check can reproduce grouped Sobol' indices for analytical 
  case from [1,2], called the Sobol' G Function.
  """

  m = SobolG()
  np.random.seed(20221103)
  m.compute_sobol_indices(100000)
  S, T, V = m.S[:,0], m.T[:,0], m.V[0]

  err = np.linalg.norm( np.round(m.Sa,2) - np.round(S,2)) + \
    np.linalg.norm( np.round(m.Ta,2) - np.round(T,2))

  print(f"\nAnalytical main effects: {np.round(m.Sa,2)}")
  print(f"Computed main effects: {np.round(S,2)}")
  print(f"\nAnalytical total effects: {np.round(m.Ta,2)}")
  print(f"Computed total effects: {np.round(T,2)}")
  assert err < 1e-1

# %%
def test_grouped_sobols_2outputs():
  """
  Check can reproduce grouped Sobol' indices for Sobol G Function
  with 2 outputs; the second output is a 2x scaling of the first. The 
  Sobol indices should be identical, and the variance should be greater
  by a factor of 4.
  """
  m = SobolG_2Outputs()
  np.random.seed(20221103)
  m.compute_sobol_indices(100000)
  S, T, V = m.S, m.T, m.V

  err = np.linalg.norm( np.round(m.Sa,2)[:,np.newaxis] - np.round(S,2)) + \
    np.linalg.norm( np.round(m.Ta,2)[:,np.newaxis] - np.round(T,2))

  err += np.abs(V[1]/V[0]-4.0)

  assert err < 1.5e-1

# %%
"""
References

[1] Sobol′, I.M. 2001. “Global Sensitivity Indices for Nonlinear Mathematical Models 
    and Their Monte Carlo Estimates.” The Second IMACS Seminar on Monte Carlo Methods 
    55 (1): 271–80. https://doi.org/10.1016/S0378-4754(00)00270-6.
[2] Saltelli, Andrea, et al. “Variance Based Sensitivity Analysis of Model Output. 
    Design and Estimator for the Total Sensitivity Index.” Computer Physics Communications, 
    vol. 181, no. 2, Feb. 2010, pp. 259–70, https://doi.org/10.1016/j.cpc.2009.09.018.
[3] Jacques, Julien, Christian Lavergne, and Nicolas Devictor. 2006. “Sensitivity 
    Analysis in Presence of Model Uncertainty and Correlated Inputs.” The Fourth 
    International Conference on Sensitivity Analysis of Model Output (SAMO 2004) 
    91 (10): 1126–34. https://doi.org/10.1016/j.ress.2005.11.047.
"""

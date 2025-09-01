#%%
import numpy as np
from DoubleQuad import *
import os, sys

sys.path.insert(0,'../')
from generalizedADE_Sobols import *

# %%
class QuadADE(DoubleQuad):
    def __init__(self, seed=20250714):

        np.random.seed(seed)
        self.instantiate_models()

        n_inner = 28
        n_outer = 28
        i1 = {
        "name": "S", 
        "dist": "uniform", 
        "group": "X_Uc",
        "params": np.array([0.2, 1.5])
        }

        i2 = {
            "name": "av_u", 
            "dist": "uniform", 
            "group": "X_Uc",
            "params": np.array([0.9, 1.1])
        }

        i3 = {
            "name": "v_p", 
            "dist": "uniform", 
            "group": "X_Uc",
            "params": np.array([0.008, 0.12])
        }

        i4 = {
            "name": "v_m", 
            "dist": "uniform", 
            "group": "X_u",
            "params": np.array([0.05, 0.15])
        }

        i5 = {
            "name": "alpha", 
            "dist": "triangular", 
            "group": "X_u",
            "params": np.array([1, 2, 1.5])
        }

        i6 = {
            "name": "lambda_s", 
            "dist": "empirical", 
            "group": "X_u_prime",
            "params": np.array([5, 1])
        }

        inputs = {
            "i1": i1, 
            "i2": i2,
            "i3": i3,
            "i4": i4, 
            "i5": i5,
            "i6": i6,    
        }
        super().__init__(inputs, n_outer, n_inner)

    def instantiate_models(self):

        N = 50000
        self.f_model = FRADESobols()
        scaling_factor = np.std(self.f_model.fX(self.f_model.get_input_samples(50000)))
        self.f_model.qoi_rescale_factor = scaling_factor

        # Using the same scaling factor for the generalizedADE. Feeding it to ctor to ensure the FRADE
        # samples for the data-consistent update are also rescaled.
        self.g_model = generalizedADEDataConsistentSobols(std_scale=1.25, qoi_rescale_factor=scaling_factor,
        make_plots=False)

        # Just sample the eigenvalues once, 
        self.DCI_eigenvalue_samples = self.g_model.get_input_samples(50000)[:,3:]

    def f(self, X_input):
        Y = self.f_model.fX(X_input)
        return Y

    def g(self, X_input):
        Y = self.g_model.fX(X_input)
        return Y

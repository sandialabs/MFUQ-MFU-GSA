# %%
import pytest
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import itertools

class AdditiveHierarchicalModel():
    """"
    Implementing the test problem 
    f(x_1, x_2) = x_1 + x_2, 

    where
    x_2 ~ N(m_o, s_o^2)
    x_1 ~ N(m, s^2), 
        for 
        m ~ N(mu, sig^2)
        s ~ exp(beta)

    here, we compare 
    1. E_{m, s}S_{x_1}
    2. S_{x_1, m, s}
    to  understand how to compute Sobol' indices for hierarchically distributed model inputs.
    """
    def __init__(self, parameter_groups=None):
        if parameter_groups is None:
            parameter_groups = [['x1'], ['x2']]
        # super().__init__(parameter_groups)
        self.parameter_groups = parameter_groups

        self.mo = 2
        self.so = 0.1

        self.beta = 2 #lam = 1/beta
        self.mu = 0
        self.sig = 1
        #self.seed = 203332
        self.seed = None
        self.n_groups = len(self.parameter_groups)
        self.get_group_indices(parameter_groups)

        self.parameter_names = [ pname for pgroup in parameter_groups for pname in pgroup ]
        self.n_params = len(self.parameter_names)

    def get_group_indices( self, parameter_groups):
      self.g_inds = [np.arange(len(self.parameter_groups[0]))]
      for group in self.parameter_groups[1:]:
        start_ind = self.g_inds[-1][-1]+1
        self.g_inds += [np.arange( start_ind, start_ind+len(group) )]

    #model f:R^p->R
    def f(self, x):
        y = x[0] + x[1]
        return y
    

    def marginal_pdf(self, H, samps):
        """
        use MC approximation of int f_{x | m, s}f_{s}f(m)dsdm
        """
        f_marg = np.zeros(samps.shape[0])
        for i in range(samps.shape[0]):
            s = 0 
            x = samps[i]
            for j in range(H.shape[0]):
                m_i = H[j, 0]
                s_i = H[j, 1]
                cond_dist = sps.norm.pdf(x, m_i, s_i)
                s = s + cond_dist
            f_marg[i] = s
        print(np.max(f_marg))
        return f_marg
    
    #sample the input space
    def get_input_samples(self, N, m_i=None, s_i=None, marg=None):
        X = np.zeros((N, 2))
        #option 2 and marginal test for opt 2
        if m_i is None or s_i is None:
            H = self.get_hyperparam_samples(N)
            if marg is not None:
                #sample from marginal on X
                np.random.seed(203332)
                N_x = 3000
                samps = np.random.uniform(-6, 6, size=N_x)
                f_marg = self.marginal_pdf(H, samps)
                u = np.random.uniform(0, 111*1/12, N_x)
                accept_reject = np.less_equal(u, f_marg)
                N_accepted = np.sum(accept_reject)
                print("N Accepted:", N_accepted)
                print("Acceptance ratio:", N_accepted/N_x)
                samps[accept_reject.reshape(N_x, )]
                X[:, 0] = samps[:N]
            else:
                #sample from joint dist
                for i in range(N):
                    m_i = H[i, 0]
                    s_i = H[i, 1]
                    X[i, 0] = np.random.normal(m_i, s_i, 1)
            X[:, 1] = np.random.normal(self.mo, self.so, N)
        #option 1
        else:
            X[:, 0] = np.random.normal(m_i, s_i, N)
            X[:, 1] = np.random.normal(self.mo, self.so, N)
        return X
    
    def get_hyperparam_samples(self, N):
        H = np.zeros((N, 2))

        if self.seed is None:
            #m
            H[:, 0] = np.random.normal(self.mu, self.sig, N)
            #s
            H[:, 1] = np.random.exponential(1/self.beta, N)
        else: 
            np.random.seed(self.seed)
            #m
            H[:, 0] = np.random.normal(self.mu, self.sig, N)
            #s
            H[:, 1] = np.random.exponential(1/self.beta, N)
        return H
    
    def compute_sobol_indices(self, N, m_i=None, s_i=None, marg=None):
      
      S = np.zeros((self.n_groups))
      T = np.zeros((self.n_groups))
      V = 0

      A = self.get_input_samples(N, m_i, s_i, marg)
      print('Matrix A')
      B = self.get_input_samples(N, m_i, s_i, marg)
      print('Matrix B')
      if A is None or B is None: return # If input sampling failed, stop here. 

      YA = np.zeros((N))
      YB = np.zeros((N))
      YCk = np.zeros((N, self.n_groups)) # Now it's indexed by the number of groups
     
      # Creating a vector to hold the current sample of the pick-and-freeze matrix.
      p_vec = np.zeros((self.n_params))
      for j in range(0, N):
        YA[j] = self.f(A[j, :])
        YB[j] = self.f(B[j, :])
        for k,inds in enumerate(self.g_inds):
          p_vec = np.copy(A[j, :])
          # Sub in B's columns for the current parameter group.
          p_vec[inds] = B[j,inds]

          YCk[j,k] = self.f(p_vec)

      mu_A = np.mean(YA)
      mu_B = np.mean(YB)
      V = (1/(2*N))*( np.sum((YA-mu_A)**2) + np.sum((YB-mu_B)**2) )

      for k in range(0,self.n_groups):
          Vk = (1/N)*np.sum(YB*(YCk[:,k]-YA))
          S[k] = Vk/ V

          # TODO: Verify this total effect index computation is correct.
          TVk = 1/(2*N) * np.sum( (YA - YCk[:,k])**2. )
          T[k] = TVk / V
      return S, T

    def plot_hyperparam_dist(self):
        N = 10000
        H = self.get_hyperparam_samples(N)

        plt.figure()
        plt.hist(H[:, 0])
        plt.title('samples of m')
        plt.show()

        plt.figure()
        plt.hist(H[:, 1])
        plt.title('samples of s')
        plt.show()
        return

    def plot_input_dist(self):
        N = 10000
        X = self.get_input_samples(N)

        plt.figure()
        plt.hist(X[:, 0])
        plt.title('samples of x_1')
        plt.show()

        plt.figure()
        plt.hist(X[:, 1])
        plt.title('samples of x_2')
        plt.show()
        return
    
    def get_expected_Sobol_index(self, N_outer):
        """
        This tests approach 1
        
        """
        H = self.get_hyperparam_samples(N_outer)
        N = 100000
        S_main = np.zeros((N_outer, 2))
        T_total = np.zeros((N_outer, 2))
        for i in range(N_outer):
            m_i = H[i, 0]
            s_i = H[i, 1]
            S_main[i, :], T_total[i, :] = self.compute_sobol_indices(N, m_i, s_i)

        S_av = np.mean(S_main, 0)
        T_av = np.mean(T_total, 0)
        return S_main, T_total, S_av, T_av

    def get_grouped_index(self, N):
        """
        this tests option 2
        """
        S_g, T_g = self.compute_sobol_indices(N)
        return S_g, T_g
    
    def get_grouped_index_marginalized(self, N):
        """
        this compares option 2 to marginalizing over m and s
        """
        S_g, T_g = self.compute_sobol_indices(N, marg=1)
        return S_g, T_g

    def plot_marginal_samps(self, X=None):
        """
        uses rejection sampling to test if we are sampling from marginal when dropping m and s 
        """
        if X is None:
            N = 1000
            X = self.get_input_samples(N, m_i=None, s_i=None, marg=1)

        X = X[:, 0]
        print('mean marg samps', np.mean(X))
        print('std marg samps', np.std(X))
        plt.figure()
        plt.hist(X.flatten())
        plt.title('X_1 marginal samps')
        return
    
    def plot_group_samps(self, X=None):
        """"
        I think will produce the same answer as plot_marginal_samps
        """
        if X is None:
            N = 10000
            X = self.get_input_samples(N, m_i=None, s_i=None, marg=None)

        X = X[:, 0]
        print('mean group samps', np.mean(X))
        print('std group samps', np.std(X))
        plt.figure()
        plt.hist(X.flatten())
        plt.title('X_1 group samples')
        return
    
    def plot_double_loop_samps(self, X=None):
        """"
        should view samples for one realization of m and s
        """
        if X is None:
            N_outer = 1
            H = self.get_hyperparam_samples(N_outer)
            N = 10000
            X_mat = np.zeros((N, N_outer))
            for i in range(N_outer):
                m_i = H[i, 0]
                s_i = H[i, 1]
                X = self.get_input_samples(N, m_i, s_i, marg=None)
                X_mat[:, i] = X[:, 0]

        plt.figure()
        plt.hist(X.flatten())
        plt.title('X_1 samples double loop')
        return
# %%
Example = AdditiveHierarchicalModel()
Example.plot_marginal_samps()

# %%
N = 1000
S_g, T_g = Example.get_grouped_index_marginalized(N)
print('the marginalized "grouped" main index for X_1 (m, and S) is \n', S_g[0])
print('the "grouped" main index for X_2 is \n', S_g[1])

print('the marginalized "grouped" total index for X_1 (m, and S) is \n', T_g[0])
print('the "grouped" total index for X_2 is \n', T_g[1])

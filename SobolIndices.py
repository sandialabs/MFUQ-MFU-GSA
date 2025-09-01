import sys
import numpy as np
from abc import ABC, abstractmethod
import pdb
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

class SobolIndices:

    # Input the problem dimension
    def __init__(self,parameter_groups, output_names=None, n_outputs=None):
      # A list of lists where each inner list is a parameter group.
      # E.g. for two parameter groups this may be 
      # parameter_groups = [['a','b'],['c','d']]
      self.parameter_groups = parameter_groups
      self.n_groups = len(self.parameter_groups) # The number of groups.

      self.get_group_indices()

      # Get the list of parameter names by looping over the parameter groups list of lists. 
      self.parameter_names = [ pname for pgroup in parameter_groups for pname in pgroup ]
      self.n_params = len(self.parameter_names)
      self.check_no_duplicate_params()

      self.output_names = output_names
      self.format_output_names()
      if not n_outputs is None:
        self.n_outputs = n_outputs
      else:
        if not output_names is None:
          self.n_outputs = len(output_names)
        else:
          self.n_outputs = 1
      if (not output_names is None) and (not n_outputs is None):
        if not len(output_names)==n_outputs:
          sys.exit(f"ERROR: The length of output_names ({len(output_names)}) does not match n_outputs ({n_outputs}).")
  
    def get_group_indices( self ):

      # Get the indices associated with each group for subsampling arrays later.
      # E.g., for 
      #   parameter_groups = [['a','b'],['c','d']] 
      # this method will return the list of indices
      # self.g_inds = [[0,1],[2,3]]

      self.g_inds = [np.arange(len(self.parameter_groups[0]))]
      for group in self.parameter_groups[1:]:
        start_ind = self.g_inds[-1][-1]+1
        self.g_inds += [np.arange( start_ind, start_ind+len(group) )]

    def set_group_indices( self, group_indices ):
      # Allows child classes to custom-set the group indices. 
      # Does some sanity checks on the provided list of group indices.
      # 
      # Inputs:
      #   group_indices: A list of lists of indices associated with each parameter group.
      #       These indices should be equal to the column for that parameter in the input 
      #       sample matrix. 
      #
      # E.g., if the input sample matrix had samples for parameters ordered like [a,b,c,d], 
      # and the parameter groups were [[b,c], [a,d]], then the child class should specify
      # the group_indices [[1,2], [0,3]]. 
      
      all_inds = np.array([ i for i_group in group_indices for i in i_group ])
      if not all_inds.size == self.n_params:
        sys.exit("ERROR: You have not specified enough indices for the number of parameters.")
      if not np.unique(all_inds).size  == self.n_params:
        sys.exit("ERROR: You used the same index for at least two parameters.")
      
      self.g_inds = group_indices

    def check_no_duplicate_params( self ):
      if not self.n_params == len(set(self.parameter_names)): 
        # set() pulls out the unique values.
        sys.exit("ERROR: A parameter is listed in more than one group! This is not permitted.")

    def format_output_names( self ):
      if not self.output_names is None and not isinstance(self.output_names,list):
        self.output_names = [self.output_names]

    # Virtual function to evaluate the model f:R^p->R
    @abstractmethod
    def f(self,x):
      # Inputs: sample vector x in R^n_params
      # Outputs: sample vector f(x) in R^n_outputs
      return
    
    # If things can be vectorize by child classes, it can and should be.
    def fX(self, X):
      # Inputs: sample vector X of size N x n_params
      # Outputs: sample vector Y of size N x n_outputs
      Y = np.zeros((X.shape[0],self.n_outputs))
      for i in tqdm(range(X.shape[0]), desc='Function evaluation'):
        Y[i,:] = self.f(X[i,:])
      return Y

    # Virtual function to sample the input space 
    @abstractmethod
    def get_input_samples(self,N):
      # Inputs: N, the number of samples to generate
      # Outputs: A NumPy array of random samples with dimension Nxp, 
      #          where p is the dimension of parameter space.
      #          The columns of parameter samples should be ordered the same as the
      #          parameter groups specified by the user. That is, if the parameter_groups
      #          list of lists was [['a','b'],['c','d']], the samples returned by this method
      #          should have columns with samples ordered as 'a', 'b', 'c', 'd'.
      return

    # Child classes can override this, but assuming it's NumPy RNG for now.
    def archive_random_state(self):
      self.random_state = np.random.get_state()

    def compute_sobol_indices(self, N=None, sample_tup=None):
      """
      This method implements the numerical computation of main and total effects Sobol'
      indices as defined in [1]. Results are stored in attributes S, T, and V for
      main effect, total effect, and variance, respectively.
      
      Inputs:
        N: The number of samples and p is the number of parameters.
        OR
        sample_tup: A tuple of independently sampled input matrices, A & B. 
      
      References:
      [1] C. Prieur and S. Tarantola, “Variance-Based Sensitivity Analysis: Theory and 
          Estimation Algorithms,” in Handbook of Uncertainty Quantification, R. Ghanem, 
          D. Higdon, and H. Owhadi, Eds. Cham: Springer International Publishing, 2017, 
          pp. 1217–1239. doi: 10.1007/978-3-319-12385-1_35.
      """

      if (N is None and sample_tup is None) or ((not N is None) and (not sample_tup is None)):
        print("ERROR: You must either pass the number of samples to generate, N,"\
              "OR a tuple of sample matrices, sample_tup.")
        return
      elif N is None:
        A, B = sample_tup
        N = A.shape[0]
      else:
        self.archive_random_state()
        A = self.get_input_samples(N)
        B = self.get_input_samples(N)
        # If input sampling failed, stop here. 
        if A is None or B is None: 
          print("Input sampling failed.")
          return 

      self.S = np.zeros((self.n_groups, self.n_outputs))
      self.T = np.zeros((self.n_groups, self.n_outputs))

      YA = self.fX(A)
      YB = self.fX(B)
      # Check if a child class implemented fX with a scalar
      # output, so that YA and YB are 1 dimensional. If so,
      # make the arrays 2D.
      if len(YA.shape)==1:
        YA = YA[:, np.newaxis]
        YB = YB[:, np.newaxis]
  
      self.V = 0.5 * (np.nanvar(YA, axis=0) + np.nanvar(YB, axis=0))


      Ab = np.zeros_like(A)
      YAb = np.zeros((N, self.n_outputs))
      for k, inds in enumerate( tqdm(self.g_inds, desc="Iteration over group")):
          # Evaluating the pick-freeze matrix
          Ab = A.copy()
          Ab[:,inds] = B[:,inds]
          YAb = self.fX(Ab)
          if len(YAb.shape)==1: YAb = YAb[:,np.newaxis]

          Vk = np.nanmean(YB*(YAb - YA), axis=0)
          self.S[k,:] = Vk / self.V

          TVk = 0.5 * np.nanmean((YA-YAb)**2, axis=0)
          self.T[k,:] = TVk / self.V
          
      self.format_sobol_results()

    def parameter_groups_to_str(self):
      param_grp_str_list = []
      for grp in self.parameter_groups:
        if len(grp) > 2:
          grp_str = ', '.join(grp[:2])
          grp_str += ', ...'
        else:
          grp_str = ', '.join(grp)
        if len(grp) > 1:
          grp_str = '{'+grp_str+'}'
        param_grp_str_list.append( grp_str )
      return param_grp_str_list

    def format_sobol_results(self):
      # Creates a dictionary with keys Main Effects, Total Effects, 
      # and Variance(s).
      #
      # Dictionary entries are Pandas DataFrames labeled by 
      # parameter name and output name, if that was set. 

      self.results = dict()
      if self.n_outputs==1:
        self.results['Main Effects'] = pd.Series( data=self.S[:,0], 
                                                index=self.parameter_groups_to_str(), 
                                                name=self.output_names)
        self.results['Total Effects'] = pd.Series( data=self.T[:,0], 
                                        index=self.parameter_groups_to_str(), 
                                        name=self.output_names)
        self.results['Variance'] = self.V[0]
      else:
        self.results['Main Effects'] = pd.DataFrame( data=self.S, 
                                                index=self.parameter_groups_to_str(), 
                                                columns=self.output_names)
        self.results['Total Effects'] = pd.DataFrame( data=self.T, 
                                        index=self.parameter_groups_to_str(), 
                                        columns=self.output_names)
        self.results['Variances'] = pd.Series( data=self.V, 
                                              index=self.output_names)

    def archive_sobol_results(self, filename):
      if not hasattr(self, 'results'): 
        sys.exit( "ERROR: You must call compute_sobol_indices"
          " before you can archive the results.") 
      
      results = [ self.random_state, self.results ]
      with open( filename, 'wb') as f:
        pickle.dump(results, f)  

    def load_sobol_results(self, filename):
      self.random_state, self.results = pickle.load(open(filename, 'rb'))

    def print_sobol_results(self):
      if not hasattr(self, 'results'): 
        sys.exit( "ERROR: You must call compute_sobol_indices"
          " before you can print its results.")
      
      for key, value in self.results.items():
        print(f"{key}:")
        print(value)

    def plot_sobol_results(self, to_plot=['Main Effects', 'Total Effects'], group_labels=None):
        if not hasattr(self, 'results'): 
            print( "ERROR: You must call compute_sobol_indices"
                   " before you can plot its results.")
            return
        
        if not isinstance(to_plot,list): to_plot = [to_plot]
        if not all(key in self.results.keys() for key in to_plot):
          print("ERROR: You need to specify a subset of the following for\n"
                f"the variable 'to_plot': {self.results.keys()}")
          return
        
        if len(to_plot)==2: width=4
        else: width=2.3
        fig = plt.figure(figsize=(width,2))
        axs = fig.subplots(1,len(to_plot),sharey=True)
        if not isinstance(axs,np.ndarray): axs=[axs]

        for ax, label in zip(axs, to_plot):
            temp = self.results[label].round(2)
            if not group_labels is None:
              temp.index = group_labels 
            temp = temp.iloc[::-1]
            temp.plot(kind='barh', ax=ax)
            ax.set_title(label)

            ax.set_xlim([0,1])
            ax.grid(axis='x')
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        return fig

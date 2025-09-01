# %%
from SobolIndices import *
from generalizedADE import *
import scipy.stats as ss
import scipy.optimize as so

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Sobol robustness source code
#################################################################

# %%
class FRADESobols(FRADE, SobolIndices):
  def __init__(self):
    super().__init__()
    parameter_groups = [["IC_mode"],["u"], ["nu_p"], ["nu_m","alpha"]]
    SobolIndices.__init__(self,parameter_groups)

  def get_input_samples( self, N ):
    X = np.zeros((N,self.n_params))
    # IC_mode
    X[:,0] = np.random.uniform(.2, 1.5, size=N )
    # u
    X[:,1] = np.random.uniform( 0.9, 1.1, size=N )
    # nu_p
    X[:,2] = np.random.uniform( .008, .012, size=N )
    # nu_m
    X[:,3] = np.random.uniform( .05, .15, size=N )
    # alpha
    X[:,4] = np.random.triangular( 1, 1.5, 2,size=N )
    return X

class FRADESobolsEachInput(FRADESobols):
  def __init__(self):
    FRADE.__init__(self)
    parameter_groups = [["IC_mode"],["u"], ["nu_p"], ["nu_m"],["alpha"]]
    SobolIndices.__init__(self,parameter_groups)

class generalizedADESobolsfromFRADE(FRADESobols):

  def __init__(self):
    FRADE.__init__(self)
    parameter_groups = [["IC_mode"], ["u"], ["nu_p"], [f"lambda{i}" for i in 1+np.arange(self.Nk-2)]]
    SobolIndices.__init__(self,parameter_groups)

  def get_input_samples(self, N):
    # Getting most samples directly from FRADE case.
    X = FRADESobols.get_input_samples(self, N).astype('complex')

    # Setting the last Nk rows to be eigenvalue samples. 
    X[:,3:] = X[:,3,np.newaxis] * self.ikx[1:-1]**X[:,4,np.newaxis]
   
    return X

  # Need to redefine we want to use the generalizedADE function because
  # we inherit from FRADE. Probably a better way to do this...
  def f_field(self, x, t=None):
    return generalizedADE.f_field(self, x, t)
  
  def f(self, x, t=None):
    return generalizedADE.f(self, x, t)
  
  def fX(self, X, t=None):
    return generalizedADE.fX(self, X, t)
  
  def set_eigenvalue_params(self, theta):
    return generalizedADE.set_eigenvalue_params(self, theta)
  
class generalizedADEMVNormalSobols(generalizedADESobolsfromFRADE):

  def __init__(self):
    super().__init__()

    eigenvalues = super().get_input_samples(100000)[:,3:]
    log_neg_r = np.log(-eigenvalues.real.T)
    log_i = np.log(eigenvalues.imag.T)
    self.log_evals_unraveled = np.concatenate([log_neg_r,log_i])
    self.mean = np.mean(self.log_evals_unraveled,axis=1)
    self.cov = np.cov(self.log_evals_unraveled,rowvar=True)

  def get_input_samples(self, N):
    # Getting most samples directly from FRADE case.
    X = FRADESobols.get_input_samples(self,N).astype('complex')

    log_eigenvals_unraveled = np.random.multivariate_normal(self.mean, self.cov, size=N)
    N_eigenvals = log_eigenvals_unraveled.shape[1]//2
    X[:,3:].real = -np.exp(log_eigenvals_unraveled[:,:N_eigenvals])
    X[:,3:].imag = np.exp(log_eigenvals_unraveled[:,N_eigenvals:])

    return X

class generalizedADEDataConsistentSobols(generalizedADEMVNormalSobols):

  """
  This class uses data-consistent inversion to generate eigenvalue samples
  that are consistent with a (possibly perturbed) distribution of QoIs drawn
  from sampling the FRADE. See [1,2] for discussion of this method. Here we use
  the rejection-sampling approach first described in [1].

  [1] Butler, T., J. Jakeman, and T. Wildey. 2018. “Combining Push-Forward Measures and Bayes’ Rule 
      to Construct Consistent Solutions to Stochastic Inverse Problems.” SIAM Journal on Scientific 
      Computing 40 (2): A984–1011. https://doi.org/10.1137/16M1087229.
  [2] Butler, Troy, T Wildey, and Tian Yu Yen. 2020. “Data-Consistent Inversion for Stochastic Input-to-Output 
      Maps.” Inverse Problems 36 (8): 085015. https://doi.org/10.1088/1361-6420/ab8f83.
  """

  def __init__(self, std_scale=1.0, qoi_rescale_factor=1, make_plots=False ):
    """
    Inputs: 
      std_scale: A factor that will be used to scale the standard deviation of the MV Normal
                 approximation for the eigenvalues used in the parent class.
      qoi_rescale_factor: A factor by which the QoI will be divided.
    """
    super().__init__()

    self.qoi_rescale_factor = qoi_rescale_factor

    self.std = np.std(self.log_evals_unraveled,axis=1)
    self.corr = np.corrcoef(self.log_evals_unraveled,rowvar=True)    

    # Scaling the standard deviation and leaving correlation alone
    # because any lower correlation introduces nonphysical oscillations
    # in the concentration field.
    self.scale_std(std_scale)
    self.compute_cov()

    self.make_plots = make_plots

    """
    Constructing Gaussian KDEs for observed and predicted QoIs.
    """ 
    #########################
    # Observed QoI KDE
    #########################
    obs_Nkde = 10000 # This was chosen heuristically such that there was good
                     # agreement between the samples and the KDE approximation
    # NOTE: For now setting the other model parameters to their
    # nominal values to get the data-consistent eigenvalue samples, 
    # but should reconsider this later.
    X = generalizedADESobolsfromFRADE.get_input_samples(self, obs_Nkde)
    X[:,0] = self.IC_mode
    X[:,1] = self.u 
    X[:,2] = self.nu_p
    q = generalizedADESobolsfromFRADE.fX(self, X)
 
    obs_support = np.linspace(np.min(q), np.max(q), 100)
    self.obs_kde = ss.gaussian_kde(q, bw_method='silverman')
    self.FRADE_samples = np.copy(q)

    if self.make_plots:
      plt.title('Observed QoI KDE vs. samples')
      plt.hist(q, histtype='step',density=True,bins=30,label='Observed samples')
      plt.hist(self.obs_kde.resample(10000)[0,:],histtype='step',bins=30,density=True,label='KDE samples')
      plt.plot(obs_support, self.obs_kde.evaluate(obs_support), label='KDE')
      plt.legend(loc='best')
      plt.show()

      print(f"Observed q variance: {np.var(q)}")
      print(f"Observed KDE variance: {np.var(self.obs_kde.resample(10000)[0,:])}")

  def scale_std(self, scale):
    self.std *= scale

  def scale_corr_off_diags( self, scale ):
    self.corr *= scale
    np.fill_diagonal(self.corr, 1.0) 

  def compute_cov(self):
    # Set the covariance matrix from standard deviations
    # and correlation matrix.
    self.cov = np.outer(self.std,self.std)*self.corr

  def get_eigenvalue_samples_with_nominals(self, N):
    # Returns an array of samples where the first three model parameters
    # are set to their nominal values and only the eigenvalue samples vary.
    X = super().get_input_samples(N)
    X[:,0] = self.IC_mode
    X[:,1] = self.u 
    X[:,2] = self.nu_p
    return X

  def get_data_consistent_eigenvalue_samples(self, N):
    """
    Produces a set of data-consistent eigenvalue samples.

    Input:
      N: The number of samples to generate before accept/reject
    Output:
      Xup: Updated set of data-consistent eigenvalue samples. There
           will be fewer than N samples unless the acceptance ratio 
           is 1. 
    """
    X = self.get_eigenvalue_samples_with_nominals(N)
    q = self.fX(X)
    q_original=np.copy(q)

    # Going ahead and screening out samples with negative concentration.
    filter_inds = np.where( (np.max(self.FRADE_samples)>=q) & (q >0))
    X = X[filter_inds]
    q = q[filter_inds]

    pred_kde = ss.gaussian_kde(q)

    if self.make_plots:
      plt.title('Predicted QoI KDE vs. samples')
      plt.hist(q, histtype='step', density=True, bins=30, label='QoI samples')
      plt.hist(pred_kde.resample(1000)[0,:], histtype='step', density=True, bins=30, label='KDE samples')
      pred_support = np.linspace(np.min(q),np.max(q),1000)
      plt.plot(pred_support,pred_kde.evaluate(pred_support),label='KDE')
      plt.legend(loc='best')
      plt.show()
      
    r = self.obs_kde.evaluate(q) / pred_kde.evaluate(q)
    print(f"Mean rejection ratio (should be >= 1): {np.nanmean(r):.2f}")

    t = np.random.uniform(size=r.size)
    accept_reject = np.less_equal( t, r/np.nanmax(r) )
    N_accepted = np.sum(accept_reject)

    Xup =  X[accept_reject]
    qup = self.fX(Xup)

    if __name__=='__main__':
      print(f"Mean rejection ratio scaled by max: {np.nanmean(r/np.nanmax(r)):.2f}")
      print("N Accepted:", N_accepted)
      print("Acceptance ratio:", N_accepted/N)
      print(f"FRADE var: {np.var(np.exp(self.FRADE_samples)):.2e}")
      print(f"Data-consistent var, nominal vars: {np.var(qup):.2e}")
    if self.make_plots:
      plt.title("Data-consistent update QoI samples")
      plt.hist(self.FRADE_samples, histtype='step', bins='auto', density=True, label='Observed samples')
      plt.hist(q_original, histtype='step', bins='auto', density=True, label='Predicted samples')
      plt.hist(qup, histtype='step', bins='auto', density=True, label='Updated predicted samples')
      plt.xlim([0,4])
      plt.legend(loc='best')
      plt.savefig('figs/data_consistent_update.pdf')
      plt.show()
    return Xup[:,3:]

  def get_input_samples(self, N):
    X = super().get_input_samples(N)

    # Have found that rejection sampling with 5x the
    # number of samples that we need, we will get enough accepted samples.
    eigenvalue_samples = self.get_data_consistent_eigenvalue_samples(5*N)
    if not eigenvalue_samples.shape[0] >= N:
      print("ERROR: Not enough data consistent samples were generated. Try again with "
                "a larger sample size for rejection sampling." )
      return # Return None if this happens; Sobol compute will stop.
    X[:,3:] = eigenvalue_samples[:N,:]
    return X

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inference source code
######################################################

# %%
import os, sys
import emcee
import numpy as np
import h5py
import scipy.stats as ss
import scipy.optimize as so
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Convenience methods for defining priors

def logn_hyperparams_from_CDF_constraints(constraint1, constraint2):
    # This method will apply two CDF constraints to derive the
    # hyperparameters of a log-normally distributed random variable.
    # It is assumed that constraints are passed as tuples where the 
    # first entry is the variable value of the constraint and the 
    # second entry is the specified probability at that variable value.
    # For example, if P(X <= 1) = 0.5, the tuple would be (1,0.5).

    # Given these constraints where we denote x1 and x2 the rv values
    # and P1, P2 the assigned probabilities, 
    #
    # sigma = ( ln(x2)-ln(x1) ) / ( PhiInv(P2) - PhiInv(P1) ),
    # where PhiInv is the inverse CDF of a standard normal RV
    #
    # mu = ln(x1) + PhiInv(P1) * sigma
    
    x1, P1 = constraint1
    x2, P2 = constraint2

    sn = ss.norm()

    sigma = (np.log(x2) - np.log(x1)) / (sn.ppf(P2) - sn.ppf(P1))
    mu = np.log(x1) - sn.ppf(P1) * sigma
    return mu, sigma

def logn_hyperparams_from_mode_and_quantile(mode, constraint_tuple):
    # This method will apply a constraint on the mode of a 
    # log-normally distributed random variable, as well as one 
    # cosntraint on the CDF, to derive the hyperparameters of the 
    # random variable. The constraint should be passed as a tuple
    # where the variable value is the first entry and the probability
    # value is the second. 
    # For example, if P(X <= 1) = 0.5, the tuple would be (1,0.5).

    # Given these constraints where we denote M the mode, xc the 
    # rv value, and Pc is the probability, 
    #
    # sigma = 0.5 * [ -PhiInv(Pc) + sqrt(PhiInv^2(Pc) - 4(ln(M)-ln(xc)))],
    # where PhiInv is the inverse CDF of a standard normal RV
    #
    # mu = ln(M)+sigma^2
    
    xc, Pc = constraint_tuple

    # sigma^2 + b sigma + c = 0 where b = PhiInv(Pc), c=ln(M)-ln(xc)
    b = ss.norm().ppf(Pc)
    c = np.log(mode) - np.log(xc)

    sigma = 0.5 * (-b + np.sqrt(b**2 - 4*c))
    mu = np.log(mode) + sigma**2
    return mu, sigma

# %%
class Bayes:
    def __init__(self, lhood_sd=1e-4, lhood_seed=20241029, lhood_type='additive'):

        self.instantiate_prior()
        self.lhood_sd = lhood_sd
        self.lhood_type = lhood_type
        self.instantiate_likelihood(lhood_seed)
        self.create_results_dir()
        
    def create_results_dir(self):
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)
        self.figdir = f"{self.results_dir}/figs/"
        if not os.path.exists(self.figdir): os.makedirs(self.figdir)

    # Abstract methods to be defined by child classes
    def instantiate_prior(self):
        return
    
    def instantiate_likelihood(self, lhood_seed):
        return
    
    def log_prior(self, theta):
        return
    
    def log_lhood(self, theta):
        return
    
    def get_prior_mean(self):
        return
    
    def get_prior_bounds(self):
        return
    
    ##############################################
    # MCMC Methods
    ##############################################
    def log_post(self, theta):
        lp = self.log_lhood(theta) + self.log_prior(theta)
        if not np.isfinite(lp): 
            return -np.inf
        return lp
    
    def run_mcmc(self, N=1000, theta0=None, restart_from_previous_run=False, seed=20240919):
        if theta0 is None:
            theta0 = self.get_prior_mean()
    
        nwalkers = 2*self.n_params
    
        # Create/load an HDF5 archive of the chain for checkpointing
        chain_archive=f'{self.results_dir}/mcmc_chain.h5'
        backend = emcee.backends.HDFBackend(chain_archive)
        
        pos = None # default is None so it uses restart.
        if not restart_from_previous_run:
            np.random.seed(seed)
            pos = theta0 + 1e-3 * np.random.randn(nwalkers, theta0.size)
            backend.reset(pos.shape[0], pos.shape[1])
    
        self.sampler = emcee.EnsembleSampler(nwalkers, theta0.size, self.log_post, backend=backend)
    
        self.sampler.run_mcmc(pos, N, progress=True);
        self.chains = self.sampler.get_chain(flat=True)
    
    def load_mcmc_chain(self, filename=None):
        if filename is None:
            filename=f'{self.results_dir}/mcmc_chain.h5'
        self.sampler = emcee.backends.HDFBackend(filename) 
        self.chains = self.sampler.get_chain(flat=True)
    
    def optimize_for_mle(self, theta0=None):
        if theta0 is None:
            theta0 = self.get_prior_mean()
        
        loss = lambda theta: - self.log_lhood(theta)

        bounds = self.get_prior_bounds()
        return so.minimize(loss, theta0, bounds=bounds)
    
    def optimize_for_map(self, theta0=None):
        if theta0 is None:
            theta0 = self.get_prior_mean()
        
        loss = lambda theta: - self.log_lhood(theta) - self.log_prior(theta)

        bounds = self.get_prior_bounds()
        return so.minimize(loss, theta0, bounds=bounds)
    
    #====================================================================
    # Plotting things 
    #====================================================================
        
    def plot_mcmc_chains(self, chains=None, return_fig=False):
        if chains is None:
            chains = self.chains
        Nrows = int(np.ceil(self.n_params/3))
        fig = plt.figure(figsize=(7,1.5*Nrows))
        axs = fig.subplots(Nrows, 3, sharex=True).flatten()
        for label, samples, ax in zip(self.parameter_names, chains.T, axs):
            ax.plot(samples)
            ax.set_xlabel('MCMC Iteration')
            ax.set_ylabel(label, rotation=0, ha='right')
            ax.spines[['top','right']].set_visible(False)

        if not self.n_params == axs.size: 
            for ax in axs[self.n_params - axs.size:]:
                fig.delaxes(ax)  
        fig.tight_layout()
        if return_fig: return fig

    def plot_marginal_histograms(self, samples, vs_prior=False, return_fig=False):
        # Inputs: 
        #   samples [N x Nparams]
        Nrows = int(np.ceil(self.n_params/3))
        fig = plt.figure(figsize=(7,1.5*Nrows))
        axs = fig.subplots(Nrows,3).flatten()

        if vs_prior:
            X = self.get_prior_samples(1000)
        for i, (label, samples, ax) in enumerate(zip(self.parameter_names, samples.T, axs)):
            ax.hist(samples, histtype='step', density=True)
            if vs_prior:
                ax.hist(X[:,i], histtype='step', density=True, label='Prior')
                if i==0:
                    ax.legend(loc='best')
            ax.set_xlabel(label)
            if i % 3 == 0: 
                ax.set_ylabel('Probability\ndensity', rotation=0, ha='right')
            ax.spines[['top','right']].set_visible(False)
        if not self.n_params == axs.size: 
            for ax in axs[self.n_params - axs.size:]:
                fig.delaxes(ax)  
        fig.tight_layout()

        if return_fig: return fig

    def plot_2d_contours(self, samples, other_samples=None, vs_prior=False, label='', return_fig=False):
        # Inputs: 
        #   samples [N x Nparams]
        #   other_samples [N x Nparams]: samples to compare to
        #   vs_prior [True/False]: if True, sets other_samples to samples from the prior.

        dim = 1.25*(self.n_params-1)
        fig = plt.figure(figsize=(dim, dim))
        axs = fig.subplots(self.n_params-1, self.n_params-1)

        if vs_prior:
            other_samples = self.get_prior_samples(1000)
            label='Posterior (black) vs Prior (blue)'
        for i, (sample1, label1) in enumerate(zip(samples[:,:-1].T, self.parameter_names[:-1])):
            for j, (sample2, label2) in enumerate(zip(samples[:,1:].T, self.parameter_names[1:])):

                ax = axs[j,i]
                if i > j:
                    plt.delaxes(ax)
                if not other_samples is None:
                    ax.plot(other_samples[:,i], other_samples[:,j+1], '.', ms=1)
                ax.plot(sample1, sample2, 'k.', ms=1)
                ax.set_xlabel(label1)
                ax.set_ylabel(label2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        if return_fig: return fig

    def plot_spatial_pushforward(self, input_samples, 
                                 t=None, 
                                 individual_samples=False, 
                                 with_noise=True, 
                                 label='',
                                 return_fig=False):
        # Inputs:
        #   input_samples [N x Nparams]
        #   t [float]: time to compute the pushforward for (default = self.t)
        #   individual_samples [True/False]: whether to plot individual samples or the 
        #   confidence interval
        #   with_noise [True/False]: whether to sample the likelihood
        #   label: the label to apply to the figure.

        if t is None:
            t = self.t

        pushforward_samples = self.get_spatial_pushforward_samples(input_samples, t, with_noise)
        N = pushforward_samples.shape[0]

        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)

        if individual_samples:
            for j in range(N):
                ax.plot(self.x, pushforward_samples[j,:], 'C0')
        else:
            q5, q95 = np.nanquantile(pushforward_samples, q=[0.05,0.95], axis=0)
            mean = np.nanmean(pushforward_samples, axis=0)
            ax.fill_between(self.x, q5, q95, color='#b4c5d1')
            ax.plot(self.x, mean)

        c_true = self.truth_model.f_field_from_eigenvalues(t=t)        
        ax.plot(self.x, c_true, 'k')

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\langle c \rangle$,'+f'\nt={t}', 
                        rotation=0, ha='right')
        ax.set_title(label)
            
        ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        if return_fig: return fig

    def plot_prior_spatial_pushforward(self, N=10, t=None, individual_samples=False, with_noise=True, return_fig=False):
        X = self.get_input_samples(N)
        self.plot_spatial_pushforward(X, t=t, individual_samples=individual_samples, with_noise=True, label='Prior Pushforward', return_fig=return_fig)

    def plot_qoi_pushforward(self, input_samples, other_samples=None, vs_prior=False, label='',  with_noise=True, return_fig=False):
        # Inputs:
        #   input_samples [N x Nparams]
        #   label: the label to apply to the figure.

        N = input_samples.shape[0]

        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)

        qoi_samples = self.get_qoi_pushforward_samples(input_samples, with_noise)
    
        ax.hist(qoi_samples, bins=30, density=True, histtype='step')

        true_qoi = self.truth_model.f_field_from_eigenvalues()[-1]
        ax.vlines(true_qoi, 0, ax.get_ylim()[1], 'k')

        if vs_prior:
            other_samples = self.get_prior_samples(N)
        
        other_label='Prior' if vs_prior else 'Other'
        if not other_samples is None:
            other_qois = self.get_qoi_pushforward_samples(other_samples, with_noise)
            ax.hist(other_qois, bins=30, density=True, histtype='step', label=other_label)
            ax.legend(loc='best')

        ax.set_xlabel(f'Concentration at outflow boundary,\n $t=${self.t:.2f}')
        ax.set_ylabel('Probability\ndensity', rotation=0, ha='right')
        ax.set_title(label)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()

        if return_fig: return fig
    
    def plot_prior_qoi_pushforward(self, N=10, return_fig=False):
        X = self.get_input_samples(N)
        self.plot_qoi_pushforward(X, label='Prior Pushforward', return_fig=return_fig)
    
    def plot_time_pushforward(self, input_samples, 
                              tvec=None, x=None, 
                              individual_samples=False, 
                              with_noise=True, 
                              vs_data=True, label='', return_fig=False):
        # Inputs:
        #   input_samples [N x Nparams]
        #   tvec [numpy array]: time to compute the pushforward for (default = self.calibration_tvec)
        #   x [float]: the x location to compute the time history at
        #   individual_samples [True/False]: whether to plot individual samples or the confidence interval
        #   with_noise [True/False]: whether to sample the likelihood
        #   label: the label to apply to the figure.

        if x is None:
            x = self.x_calibration
            # TODO: add a loop here if x is more than a scalar?
        if tvec is None:
            tvec = self.calibration_tvec.copy()

        N = input_samples.shape[0]
        pushforward_samples = self.get_time_pushforward_samples(input_samples,tvec,x,with_noise)

        c_true = self.truth_model.f_time_field_from_eigenvalues(tvec=tvec, x=x)

        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
            
        if individual_samples:
            for j in range(N):
                ax.plot(tvec, pushforward_samples[j,:], 'C0')
        else:
            q1,q99 = np.nanquantile(pushforward_samples, q=[0.01,0.99], axis=0)
            mean = np.nanmean(pushforward_samples, axis=0)
            ax.fill_between(tvec, q1, q99, color='#b4c5d1')
            ax.plot(tvec, mean)

        if vs_data:        
            ax.plot(tvec, c_true, 'k')

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\langle c \rangle$,'+f'\nx={x:.2f}', 
                        rotation=0, ha='right')
        ax.set_title(label)

        ax.spines[['top','right']].set_visible(False)

        fig.tight_layout()

        if return_fig: return fig
    
    def plot_prior_time_pushforward(self, N, tvec=None, x=None, 
                                    individual_samples=False, with_noise=True, vs_data=True, return_fig=False):
        X = self.get_input_samples(N)
        self.plot_time_pushforward(X, tvec=tvec, x=x, individual_samples=individual_samples, with_noise=with_noise, vs_data=vs_data, label='Prior Pushforward', return_fig=return_fig)

    #=============================================================
    # Predictive sampling 
    #=============================================================

    def get_spatial_pushforward_samples(self, 
                                        input_samples, 
                                        t=None, 
                                        with_noise=True):
           
        if t is None:
            t = self.t

        N = input_samples.shape[0]
        pushforward_samples = np.zeros((N, self.Nx)) 
        for j in range(N):
            pushforward_samples[j,:] = self.f_field(input_samples[j,:], t=t)
        if with_noise:
            if self.lhood_type=='multiplicative':
                pushforward_samples *= np.exp(ss.multivariate_normal(np.zeros_like(self.x), 
                                                             cov=self.lhood_sd**2.).rvs(N))
            else:
                pushforward_samples += ss.multivariate_normal(np.zeros_like(self.x), 
                                                             cov=self.lhood_sd**2.).rvs(N)
        return pushforward_samples
    
    def get_time_pushforward_samples(self, 
                                     input_samples, 
                                     tvec=None, x=None, 
                                     with_noise=True):
        
        if x is None:
            x = self.x_calibration 
            self.check_x_location(x)
        if tvec is None:
            tvec = self.calibration_tvec.copy()

        N = input_samples.shape[0]
        pushforward_samples = np.zeros((N,tvec.size))
        for j in range(N):
            pushforward_samples[j,:] = self.f_time_field(input_samples[j,:], tvec=tvec, x=x)

        if with_noise:
            noise_rv = ss.multivariate_normal(np.zeros(tvec.size), cov=self.lhood_sd**2.)
            noise_samples = noise_rv.rvs(N)
            if self.lhood_type=='multiplicative':
                pushforward_samples *= np.exp(noise_samples)
            else:
                pushforward_samples += noise_samples
        return pushforward_samples

    def get_qoi_pushforward_samples(self, 
                                    input_samples, 
                                    with_noise=True):
        N = input_samples.shape[0]
        qoi_samples = np.zeros(N)
        for j in range(N):
            qoi_samples[j] = self.f(input_samples[j,:]) 

        if with_noise:
            noise_rv = ss.norm(0,scale=self.lhood_sd)
            noise_samples = noise_rv.rvs(N)
            if self.lhood_type=='multiplicative':
                qoi_samples *= np.exp(noise_samples)
            else:
                qoi_samples += noise_samples
        return qoi_samples

    #=============================================================
    # Other postprocessing
    #=============================================================
    def compute_correlations(self, samples):
        df = pd.DataFrame(data=samples, columns=self.parameter_names)
        return df.corr()


# %%
# Complex fractional MFU
# Assumes triangular PDF for alphas
# Assumes time series data

class cFRADEMFU( cFRADE, SobolIndices, Bayes):
    # Calibration happens for a short time series at a specific x location.
    # The prediction QoI is the breakthrough time at a downstream x location.

    def __init__(self, truth_model, 
                 x_calibration=1.3984375, 
                 t_calibration=0.2,
                 Nt_calibration=10, 
                 lhood_sd=1e-4, 
                 lhood_seed=20241029,
                 lhood_type='multiplicative',
                 results_dir='results/Bayes/cFRADEMFU'):

        self.truth_model = truth_model
        FRADE.__init__(self, truth_model.t, truth_model.x_qoi) # Instantiates numerical solver

        # The x location at which the breakthrough time
        # is computed
        self.x_qoi = truth_model.x_qoi

        # The x location and time horizon of the time
        # sequence used for calibration
        self.x_calibration = x_calibration
        self.t_calibration = t_calibration
        self.Nt_calibration = Nt_calibration

        self.define_params()
        SobolIndices.__init__(self, self.parameter_groups)

        self.sample_from_posterior = False
        self.lhood_type=lhood_type
        self.results_dir=results_dir
        Bayes.__init__(self, lhood_sd, lhood_seed, lhood_type)

#===========================================================
# Class instantiation things
#===========================================================
    def define_params(self):
        model_params = ["IC_mode", "u", "nu_p"]
        MFU_params = [ "nur", 'nui', 'alphar', "alphai" ]
        MFU_hyperparams = [ "nu_mr", "nu_mi", "nu_sr", "nu_si", "alpha_cr", "alpha_ci" ]
        self.parameter_groups = [model_params, MFU_hyperparams+MFU_params]
        self.n_MFU_hyperparams=len(MFU_hyperparams)
        self.n_MFU_params=len(MFU_params)
        self.n_model_params = len(model_params)
        self.n_params = self.n_model_params + self.n_MFU_params + self.n_MFU_hyperparams
        self.n_input_params = self.n_model_params + self.n_MFU_params

        # Used later for indexing parameters for model evaluation
        self.hyperparam_inds = self.n_model_params + np.arange(self.n_MFU_hyperparams)

    def instantiate_prior(self):

        # mode = nominal, 95% support below 1.2 * nominal
        mu, sigma = logn_hyperparams_from_mode_and_quantile(1, (1.2,0.95))
        self.IC_mode_prior = ss.lognorm(scale=np.exp(mu), s=sigma)
        self.u_prior = ss.lognorm(scale=np.exp(mu), s=sigma)

        mu, sigma = logn_hyperparams_from_mode_and_quantile(0.01, (0.012,0.95))
        self.nu_p_prior = ss.lognorm(scale=np.exp(mu), s=sigma)

        self.model_param_priors = [self.IC_mode_prior, 
                                   self.u_prior, 
                                   self.nu_p_prior
                                   ]

        self.alphar_prior = lambda c: ss.triang(loc=1, scale=1, c=c) # makes a triangular distribution on [1,2] with a mode at 1+c.
        self.alphai_prior = lambda c: ss.triang(loc=1, scale=1, c=c) 
        self.nur_prior = lambda m,s: ss.lognorm(scale=np.exp(m), s=s)
        self.nui_prior = lambda m,s: ss.lognorm(scale=np.exp(m), s=s)
        self.MFU_param_priors = [self.nur_prior, self.nui_prior, self.alphar_prior, self.alphai_prior]

        # Setting hyperpriors for mu and sigma of nu
        mu_nu, sigma_nu = logn_hyperparams_from_CDF_constraints((0.1,0.1),(0.5,0.99))
        
        self.nu_mr_prior = ss.norm(loc=mu_nu, scale=.5*np.abs(mu_nu))
        self.nu_mi_prior = ss.norm(loc=mu_nu, scale=.5*np.abs(mu_nu))
        mu, sigma = logn_hyperparams_from_mode_and_quantile(sigma_nu, (1.5*sigma_nu,0.99))
        self.nu_sr_prior = ss.lognorm(scale=np.exp(mu), s=sigma)
        self.nu_si_prior = ss.lognorm(scale=np.exp(mu), s=sigma)

        self.alpha_cr_prior = ss.uniform(0,1)
        self.alpha_ci_prior = ss.uniform(0,1)

        self.MFU_hyperpriors = [ self.nu_mr_prior, 
                                 self.nu_mi_prior, 
                                 self.nu_sr_prior, 
                                 self.nu_si_prior, 
                                 self.alpha_cr_prior,
                                 self.alpha_ci_prior ]

        self.priors = self.model_param_priors+self.MFU_hyperpriors
        self.hierarchical_priors = [self.nu_mr_prior, self.nu_mi_prior, self.alphar_prior, self.alphai_prior]

    def instantiate_likelihood(self, lhood_seed):
        # Here we reduce to the x location and time series that we want
        # to use for calibration.

        self.calibration_tvec = np.linspace(0, self.t_calibration, self.Nt_calibration+1, endpoint=True)

        # Removing the 0 timestep because can have very small negative concentrations at t=0 due to truncation 
        # error in the Fourier solution.
        if self.lhood_type=='multiplicative':
            self.calibration_tvec = self.calibration_tvec[1:]
            self.calibration_data = np.log(self.truth_model.f_time_field_from_eigenvalues(tvec=self.calibration_tvec, 
                                                                                          x=self.x_calibration).flatten(order='F'))
        else:
            self.calibration_data = self.truth_model.f_time_field_from_eigenvalues(tvec=self.calibration_tvec, 
                                                                                   x=self.x_calibration).flatten(order='F')
        self.likelihood = ss.multivariate_normal(np.zeros_like(self.calibration_data), cov=self.lhood_sd**2.)

        np.random.seed(lhood_seed)
        self.true_evolution = self.calibration_data.copy()
        self.calibration_data += self.likelihood.rvs()

#====================================================================
# Sampling methods
#====================================================================

    def get_prior_samples(self, N, reject_negative=False):
        X = np.zeros((N,self.n_params))
        for i, prior in enumerate(self.priors):     
            X[:,i] = prior.rvs(N)
        hyperparam_samples = X[:,self.n_model_params:-self.n_MFU_params]
        X[:,-self.n_MFU_params:] = self.get_hierarchical_samples(hyperparam_samples)
        if reject_negative:
            X = self.filter_out_negative_concentration_samples(X)
        return X
    
    def get_hierarchical_samples(self, hyperparam_samples):
        # Given samples of hyperparameters, sample the parameter distributions
        X = np.zeros((hyperparam_samples.shape[0],self.n_MFU_params))

        # Note: this is currently hard-coded assuming 2 MFU parameters.
        for i, (nu_mr, nu_mi, nu_sr, nu_si, alpha_cr, alpha_ci) in enumerate(hyperparam_samples):
            # nu_mr
            X[i, 0] = self.nur_prior(nu_mr, nu_sr).rvs()
            X[i, 1] = self.nui_prior(nu_mi, nu_si).rvs()
            X[i, 2] = self.alphar_prior(alpha_cr).rvs()
            X[i, 3] = self.alphar_prior(alpha_ci).rvs()
        return X 
    
    def get_input_samples( self, N ):
        if not self.sample_from_posterior:
            return self.get_prior_samples(N)
        elif not hasattr(self, 'posterior_kde'):
            print("ERROR, you have specified to sample from the posterior, but you have not defined a posterior KDE") 
        else:
            return self.posterior_kde.resample(N).T

    def get_posterior_pushforward_samples(self, full_chain_samples, reject_negative=False):
        # Given the samples from the full joint posterior (including MFU parameters), 
        # replace the MFU parameter samples in the chain with resampled values

        temp = full_chain_samples.copy()
        hyperparam_samples = full_chain_samples[:, self.n_model_params:-self.n_MFU_params]
        temp[:,-self.n_MFU_params:] = self.get_hierarchical_samples(hyperparam_samples)

        if reject_negative:
            temp = self.filter_out_negative_concentration_samples(temp)

        return temp
    
    def filter_out_negative_concentration_samples(self, samples):
        positive_flag = np.full(samples.shape[0], True, dtype='bool')
        for i, theta in enumerate(samples):
            for t in [0.1,0.5,1,1.5,2.0]:
                f = self.f_field(theta, t=t)
                if np.any(f < -1e-13):
                    positive_flag[i] = False
        return samples[positive_flag]


#====================================================================
# Model evaluation methods
#====================================================================
    def get_theta_for_eval(self,theta):
        if theta.size == self.n_input_params:
            return theta
        else:
            return np.delete(theta,self.hyperparam_inds)
    
    def f_field(self, theta, t=None):
            return super().f_field(self.get_theta_for_eval(theta), t)
    
    def f_time_field(self, theta, tvec=None, x=None):
        return super().f_time_field(self.get_theta_for_eval(theta), tvec, x)
    
    def f(self, theta):
        return super().f(self.get_theta_for_eval(theta))
    
    def fX(self, X):
        Y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            Y[i] = self.f(X[i])
        return Y

#====================================================================
# Bayesian things
#====================================================================
    def log_prior(self, theta):
    
            # model parameters and hyperparameters
            log_prior = 0
            for prior, t in zip(self.priors, theta[:-self.n_MFU_params]):
                log_prior += prior.logpdf(t)
    
            # hierarchical coefficient priors
            nu_mr, nu_mi, nu_sr, nu_si, alpha_cr, alpha_ci = theta[self.n_model_params:-self.n_MFU_params]
            log_prior += self.nur_prior(nu_mr, nu_sr).logpdf(theta[-4]) 
            log_prior += self.nui_prior(nu_mi, nu_si).logpdf(theta[-3]) 
            log_prior += self.alphar_prior(alpha_cr).logpdf(theta[-2]) 
            log_prior += self.alphai_prior(alpha_ci).logpdf(theta[-1]) 
            
            return log_prior
    
    def log_lhood(self, theta):
        for t in [0.1,0.5,1,1.5,2.0]:
            f = self.f_field(theta, t=t)
            if np.any(f < -1e-13):
                return -np.inf
        try:
            if self.lhood_type=='multiplicative':
                f = np.log(self.f_time_field(theta, self.calibration_tvec, self.x_calibration).flatten(order='F'))
            else:
                f = self.f_time_field(theta, self.calibration_tvec, self.x_calibration).flatten(order='F')
            log_lhood = self.likelihood.logpdf(self.calibration_data-f)
            return log_lhood 
        except:
            return -np.inf
    
    def get_prior_mean(self):
        theta0 = np.zeros(self.n_params)
        # Getting mean of model parameter priors and hyperpriors
        #for i, prior in enumerate(self.priors[:-self.n_MFU_params]):
        for i, prior in enumerate(self.priors):
            theta0[i] = prior.mean()
        # Given the mean of the hyperpriors, getting the mean of the coefficient
        # priors
    
        nu_mr, nu_mi, nu_sr, nu_si, alpha_cr, alpha_ci = theta0[self.n_model_params:-self.n_MFU_params]
        theta0[-4] = self.nur_prior(nu_mr,nu_sr).mean()
        theta0[-3] = self.nui_prior(nu_mi,nu_si).mean()
        theta0[-2] = self.alphar_prior(alpha_cr).mean()
        theta0[-1] = self.alphai_prior(alpha_ci).mean()
        return theta0
    
    def get_prior_bounds(self):
        # For model parameter priors and hyperpriors
        bounds = [ (prior.ppf(0.05), prior.ppf(0.95)) for prior in self.priors ]

        # hardcoding the ones for the MFU parameters for now
        bounds += [ (0,np.inf), (0,np.inf), (1,2), (1,2)]
        return bounds 

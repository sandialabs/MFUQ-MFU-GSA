import numpy as np 
import os, sys
from scipy.special import roots_legendre
from scipy.special import roots_hermite
from abc import ABC, abstractmethod
from Quad import *

class DoubleQuad(Quad):

    #define inputs 
    def __init__(self, inputs, n_outer, n_inner):
        self.inputs = inputs
        self.n_outer = n_outer
        self.n_inner = n_inner
        super().__init__()
    
    # Virtual function to evaluate the model q:R^p->D
    @abstractmethod
    def f(self, x):
        return
    
    # Virtual function to evaluate the model f:R^p->D
    @abstractmethod
    def g(self, x):
        return
    
    @abstractmethod
    def get_empirical_samples(self):
        return
    
    @abstractmethod
    def get_N(self):
        return
    
    def Exp_f_squared(self, fN, W, weight_factor):
        E_f_2 = self.quad_integration(fN, W)
        E_f_2 = weight_factor * E_f_2
        return E_f_2
    
    def Exp_f(self, fN, W, weight_factor):
        E_f = self.quad_integration(fN, W)
        E_f = weight_factor * E_f
        return E_f
    
    def cond_variance_fun(self, f, N, W, weight_factor):
        fN = f(N)
        E_f_2 = self.Exp_f_squared(fN, W, weight_factor)
        E_f = self.Exp_f(fN, W, weight_factor)
        E_2_f = E_f * E_f
        V = E_f_2 - E_2_f
        return V
    
    def settup_outer_int_X_u_c(self):
    
        index_list = []
        key_list = []
        weight_factor_list = []

        for index, key in enumerate(self.inputs):
            if self.inputs[key]["group"] == "X_Uc":
                index_list.append(index)
                key_list.append(key)

        nodes_outer, weights_outer = roots_legendre(self.n_outer)

        dim_X_Uc = len(index_list)
        nodes_mat = np.zeros((dim_X_Uc, self.n_outer))
        weight_mat = np.zeros((dim_X_Uc, self.n_outer))
        
        for i in range (0, len(index_list)):
            k = key_list[i]
            if self.inputs[k]["dist"] == "uniform":
                a = self.inputs[k]["params"][0]
                b = self.inputs[k]["params"][1]
                wf = 1/(b-a)
                weight_factor_list.append(wf)

                nodes_mat[i, :] = self.limit_fun(nodes_outer, a, b)
                weight_mat[i, :] = self.weight_fun(weights_outer, a, b)

        X_outer, W_outer = self.get_tensor_product(nodes_mat, weight_mat, dim_X_Uc)
    

        return X_outer, W_outer, weight_factor_list
    
    def setup_inner_X_u(self):

        index_list_u = []
        key_list_u = []
        weight_factor_list = []
        
        nodes_inner_L, weights_inner_L = roots_legendre(self.n_inner)

        for index, key in enumerate(self.inputs):
            if self.inputs[key]["group"] == "X_u":
                index_list_u.append(index)
                key_list_u.append(key)

        
        dim_X_U = len(index_list_u)
        nodes_mat_u = np.zeros((dim_X_U, self.n_inner))
        weight_mat_u = np.zeros((dim_X_U, self.n_inner))

        for i in range (0, len(index_list_u)):
            k = key_list_u[i]
            if self.inputs[k]["dist"] == "uniform":
                a = self.inputs[k]["params"][0]
                b = self.inputs[k]["params"][1]
                wf = 1/(b-a)
                weight_factor_list.append(wf)

                nodes_mat_u[i, :] = self.limit_fun(nodes_inner_L, a, b)
                weight_mat_u[i, :] = self.weight_fun(weights_inner_L, a, b) 

            if self.inputs[k]["dist"] == "triangular":
                a = self.inputs[k]["params"][0]
                b = self.inputs[k]["params"][1]
                mode = self.inputs[k]["params"][2]
                wf = 1
                weight_factor_list.append(wf)
                
                nodes_triang = self.limit_fun(nodes_inner_L, a, b)
                nodes_mat_u[i, :] = nodes_triang
                weight_mat_u[i, :] = self.weight_fun(weights_inner_L, a, b) * self.pdf_triangle(nodes_triang, b, mode)

        X_inner_u, W_inner_u = self.get_tensor_product(nodes_mat_u, weight_mat_u, dim_X_U)

        return X_inner_u, W_inner_u, weight_factor_list
    
    def setup_inner_X_u_prime(self):

        index_list_u = []
        key_list_u = []
        weight_factor_list = []
        
        X_u_prime_inputs = [ input for input in self.inputs.values() if input["group"]=="X_u_prime"]
        if len([input for input in X_u_prime_inputs if input['dist']=='empirical']) > 0: 
        
            X_inner_u_prime = self.DCI_eigenvalue_samples.copy()

            N = X_inner_u_prime.shape[0]
            W_inner_u_prime = 1/N * np.ones(N,)

            wf = 1
            weight_factor_list.append(wf)

        else:
            nodes_inner_H, weights_inner_H = roots_hermite(self.n_inner)
            #here we change nodes and weights   
            #should import samples and assign weights as 1/length(samples)

            for index, key in enumerate(self.inputs):
                if self.inputs[key]["group"] == "X_u_prime":
                    index_list_u.append(index)
                    key_list_u.append(key)

            dim_X_U = len(index_list_u)
            nodes_mat_u = np.zeros((dim_X_U, self.n_inner))
            weight_mat_u = np.zeros((dim_X_U, self.n_inner))

            for i in range (0, len(index_list_u)):
                k = key_list_u[i]        
                if self.inputs[k]["dist"] == "normal":
                    mu = self.inputs[k]["params"][0]
                    sigma = self.inputs[k]["params"][1]
                    nodes_mat_u[i, :] = self.hermite_change_unit(nodes_inner_H, mu, sigma)
                    weight_mat_u[i, :] = weights_inner_H 
                    wf = (1/np.sqrt(np.pi))
                    weight_factor_list.append(wf)

            #change this to a matrix output of samples and remove above 
            #weight factor list should be 1
            X_inner_u_prime, W_inner_u_prime = self.get_tensor_product(nodes_mat_u, weight_mat_u, dim_X_U)

        return X_inner_u_prime, W_inner_u_prime, weight_factor_list

#===================================================#
  # Inner loop integration
#===================================================#
    def inner_loop_integration(self, X_Uc):

        #step 2 X_u 
        X_inner_u, W_inner_u, weight_factor_list_u = self.setup_inner_X_u()
        wf_u = np.prod(weight_factor_list_u)
        ####### compute var_X_U (f(x) | X_Uc) #########
        #we need an input matrix for f so we need the X_Uc^(i) samples to be repmatted
        x_uc = X_Uc * np.ones((np.size(W_inner_u), len(X_Uc)))
        X_input1 = np.hstack((x_uc, X_inner_u))
        V_Xu = self.cond_variance_fun(self.f, X_input1, W_inner_u, wf_u)

        #step 3 X_u_'
        X_inner_u_prime, W_inner_u_prime, weight_factor_list_up = self.setup_inner_X_u_prime()
        wf_up = np.prod(weight_factor_list_up)
        ######## compute var_X'_U' (g(x) | X_Uc) ##########
        x_uc2 = X_Uc * np.ones((np.size(W_inner_u_prime), len(X_Uc)))
        X_input3 = np.hstack((x_uc2, X_inner_u_prime))
        V_Xu_prime = self.cond_variance_fun(self.g, X_input3, W_inner_u_prime, wf_up)
        
        #compute g(X_u^(i))
        h_X_u = V_Xu - V_Xu_prime
        G_X_u_c_i = abs(h_X_u)
        
        return G_X_u_c_i
    
#===================================================#
# Outer loop integration
#===================================================#
    def outer_loop_integration(self):

        X_outer, W_outer, weight_factor_list_o = self.settup_outer_int_X_u_c()
        wf = np.prod(weight_factor_list_o)

        #Go to inner loop with these outer looop samples
        inner_samps = np.zeros((np.size(W_outer), 1))
        for i in range(0, np.size(W_outer)):
            print("Outer loop % done:",i/np.size(W_outer)*100)
            X_Uc = X_outer[i, :]
            inner_samps[i] = self.inner_loop_integration(X_Uc)
        
        inner_samps = inner_samps.ravel()
        E = self.quad_integration(inner_samps, W_outer)
        E = wf * E
        return E

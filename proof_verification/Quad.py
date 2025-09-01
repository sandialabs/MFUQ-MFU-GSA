import numpy as np 

class Quad:

    #define inputs 
    def __init__(self):
        return 
        
    def pdf_triangle(self, x, b, m):
        return ((b - m) - abs(m - x))/( (b - m)**2 )
    
    def limit_fun(self, x, a, b):
        return ( (b-a)/2 * x + (b+a)/2 )

    def weight_fun(self, w, a, b):
        return (b-a)/2 * w

    def hermite_change_unit(self, nodes, m, sigma):
        y = np.sqrt(2)*sigma*nodes + m
        return y
    
    def get_tensor_product(self, node_mat, weight_mat, dim_param):
   
        if dim_param == 3:
            n1 = node_mat[0, :]
            n2 = node_mat[1, :]
            n3 = node_mat[2, :]

            T1, T2, T3 = np.meshgrid(n1, n2, n3, indexing="ij")
            X = np.array([T1.flatten(), T2.flatten(), T3.flatten()])

            w1 = weight_mat[0, :]
            w2 = weight_mat[1, :]
            w3 = weight_mat[2, :]

            nu_1, nu_2, nu_3 = np.meshgrid(w1, w2, w3, indexing='ij')
            nu_mat = nu_1 * nu_2 * nu_3
            W = nu_mat.flatten()

        if dim_param == 2:
            n1 = node_mat[0, :]
            n2 = node_mat[1, :]
            
            T1, T2 = np.meshgrid(n1, n2, indexing="ij")
            X = np.array([T1.flatten(), T2.flatten()])

            w1 = weight_mat[0, :]
            w2 = weight_mat[1, :]

            nu_1, nu_2 = np.meshgrid(w1, w2, indexing='ij')
            nu_mat = nu_1 * nu_2 
            W = nu_mat.flatten()

        if dim_param == 1: 
            n1 = node_mat[0, :]
            X = np.array([n1.flatten()])

            w1 = weight_mat[0, :]
            W = w1.flatten()

        X = np.transpose(X)
        return X, W
    
    def quad_integration(self, samples, weights):
        prod1 = weights * samples
        s1 = sum(prod1)
        return s1
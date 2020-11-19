import numpy as np

def get_nercome_covariance(x, s, ndraws=500):
    ''' Computes the covariance matrix using the
        Non-parametric Eigenvalue-Regularized COvariance Matrix Estimator
        (NERCOME) described in Joachimi 2017
  
        Input
        -----
        x : np.array with shape (n_realisations, n_variables)
        s : index (between 0 and n_realisations-1) where to divide the realisations
        ndraws: number of random realisations of the partitions

        Returns
        -----
        cov_nercome : the NERCOME covariance matrix
    '''
    nreal, nbins = x.shape
    assert s < nreal
    idx = np.arange(nreal)
    
    #-- Compute the matrix many times and then average it
    cov_nercome = np.zeros((nbins, nbins))
    for i in range(ndraws):
        #-- Randomly divide all realisations in two batches
        choice = np.random.choice(idx, size=s, replace=False)
        selection_1 = np.in1d(idx, choice)
        selection_2 = ~selection_1
        x1 = x[selection_1]
        x2 = x[selection_2]
        #-- Estimate sample covariance matrices of the two samples
        cov_sample_1 = np.cov(x1.T)
        cov_sample_2 = np.cov(x2.T)
        #-- Extract eigen values and vectors (we only use vectors)
        #-- which make the U_1 matrix in Eq. 2
        eigvals_1, eigvects_1 = np.linalg.eigh(cov_sample_1)
        #-- Evaluating Eq. 2:
        #-- first U_1^T S_2 U_1
        mid_matrix = eigvects_1.T @ cov_sample_2 @ eigvects_1
        #-- make it a diagonal matrix
        mid_matrix_diag = np.diag(np.diag(mid_matrix))
        #-- now compute Z = U_1 diag( U_1^T S_2 U_1 ) U_1^T
        cov_nercome += eigvects_1 @ mid_matrix_diag @ eigvects_1.T
    cov_nercome /= ndraws
    return cov_nercome


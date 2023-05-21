import numpy as np
import scipy
cov = np.corrcoef(UX.T)
remove_idx = []

cov_new    = np.delete(cov, remove_idx, axis=0)
cov_new    = np.delete(cov_new, remove_idx, axis=1)

L          = scipy.linalg.cholesky(cov_new, lower=True)

cov_new = []
cov     = []
def mask_generation(mb_size_, pi_):
    '''
        Phi(x; mu, sigma) = 1/2 * (1 + erf( (x-mu)/(sigma * sqrt(2)) ))
        --> Phi(x; 0,1)   = 1/2 * (1 + erf( x/sqrt(2) ))
    '''
    if len(remove_idx) == 0:
        epsilon = np.random.normal(loc=0., scale=1., size=[np.shape(L)[0], mb_size_])
        g       = np.matmul(L, epsilon)
    else:
        present_idx = [i for i in range(x_dim) if i not in remove_idx]
        epsilon     = np.random.normal(loc=0., scale=1., size=[np.shape(L)[0], mb_size_])
        g2      = np.random.normal(loc=0., scale=1., size=[len(remove_idx), mb_size_])
        g1      = np.matmul(L, epsilon)
        g       = np.zeros([x_dim, mb_size_])

        g[present_idx, :] = g1
        g[remove_idx, :]  = g2

    m = (1/2 * (1 + scipy.special.erf(g/np.sqrt(2)) ) < pi_).astype(float).T
    return m

m_mb = mask_generation(x_mb.shape[0], p)

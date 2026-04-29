import numpy as np
from sklearn.mixture import GaussianMixture as SklearnGMM
from ..core.distributions import Gaussian, GaussianMixture

def fit_gmm_to_data(data, n_components=1, covariance_type='full', max_iter=100, random_state=None):
    """
    Fits a Gaussian Mixture Model to empirical data using Expectation-Maximization.
    
    Args:
        data: np.ndarray of shape (n_samples, n_features)
        n_components: int, number of Gaussian components (N)
        
    Returns:
        GaussianMixture: Our custom GMM object representing the fitted potential.
    """
    data = np.atleast_2d(data)
    
    # Delegate EM algorithm to scikit-learn for numerical stability and performance
    sklearn_gmm = SklearnGMM(
        n_components=n_components, 
        covariance_type=covariance_type, 
        max_iter=max_iter,
        random_state=random_state
    )
    
    sklearn_gmm.fit(data)
    
    components = []
    for i in range(n_components):
        mean = sklearn_gmm.means_[i]
        
        # Scikit-learn covariance shapes depend on the covariance_type
        if covariance_type == 'full':
            cov = sklearn_gmm.covariances_[i]
        elif covariance_type == 'diag':
            cov = np.diag(sklearn_gmm.covariances_[i])
        elif covariance_type == 'spherical':
            cov = np.eye(len(mean)) * sklearn_gmm.covariances_[i]
        elif covariance_type == 'tied':
            cov = sklearn_gmm.covariances_
            
        components.append(Gaussian(mean, cov))
        
    return GaussianMixture(sklearn_gmm.weights_, components)

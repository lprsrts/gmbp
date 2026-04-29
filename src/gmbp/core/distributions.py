import numpy as np
from scipy.stats import multivariate_normal

class Gaussian:
    """Represents a Multivariate Gaussian distribution."""
    
    def __init__(self, mean, cov):
        self.mean = np.atleast_1d(mean).astype(float)
        self.cov = np.atleast_2d(cov).astype(float)
        
        # Canonical form parameters (Information form) 
        # Crucial for stable Belief Propagation multiplications.
        self.precision = np.linalg.inv(self.cov)
        self.information = self.precision @ self.mean
        
        self.dim = self.mean.shape[0]

    def __mul__(self, other):
        """
        Multiplies two Gaussians. 
        In canonical form, multiplication is simply the addition of information vectors and precision matrices.
        """
        if not isinstance(other, Gaussian):
            raise TypeError("Can only multiply Gaussian with another Gaussian.")
            
        if self.dim != other.dim:
            raise ValueError("Gaussians must have the same dimensionality to be multiplied.")

        new_precision = self.precision + other.precision
        new_cov = np.linalg.inv(new_precision)
        new_info = self.information + other.information
        new_mean = new_cov @ new_info
        
        # Calculate the scaling factor (the integral of the product)
        # This becomes the new weight when multiplying mixtures.
        c = multivariate_normal.pdf(self.mean, mean=other.mean, cov=self.cov + other.cov)

        return Gaussian(new_mean, new_cov), c

    def __repr__(self):
        return f"Gaussian(dim={self.dim}, mean={self.mean})"


class GaussianMixture:
    """Represents a Gaussian Mixture Model (GMM)."""
    
    def __init__(self, weights, components):
        if len(weights) != len(components):
            raise ValueError("Number of weights must match number of components.")
            
        self.weights = np.array(weights).astype(float)
        # Normalize weights to ensure they sum to 1
        self.weights /= np.sum(self.weights)
        self.components = components
        self.dim = components[0].dim

    def __mul__(self, other):
        """
        Multiplies two Gaussian Mixtures.
        Results in a new mixture with N * M components.
        """
        if not isinstance(other, GaussianMixture):
            raise TypeError("Can only multiply GaussianMixture with another GaussianMixture.")
            
        new_weights = []
        new_components = []

        for w1, comp1 in zip(self.weights, self.components):
            for w2, comp2 in zip(other.weights, other.components):
                # Product of two Gaussians
                new_comp, scale = comp1 * comp2
                
                # The new weight is the product of prior weights and the scaling factor
                new_weight = w1 * w2 * scale
                
                new_weights.append(new_weight)
                new_components.append(new_comp)

        return GaussianMixture(new_weights, new_components)

    def prune(self, threshold=1e-5):
        """Removes components with weights below the threshold and renormalizes."""
        kept_weights = []
        kept_components = []
        
        for w, comp in zip(self.weights, self.components):
            if w >= threshold:
                kept_weights.append(w)
                kept_components.append(comp)
                
        self.weights = np.array(kept_weights)
        self.weights /= np.sum(self.weights)
        self.components = kept_components

    def __repr__(self):
        return f"GaussianMixture(components={len(self.components)}, dim={self.dim})"

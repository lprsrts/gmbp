import numpy as np
from scipy.stats import multivariate_normal

class Gaussian:
    def __init__(self, mean, cov):
        self.mean = np.atleast_1d(mean).astype(float)
        self.cov = np.atleast_2d(cov).astype(float)
        self.cov = (self.cov + self.cov.T) / 2.0
        
        try:
            self.precision = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            self.precision = np.linalg.pinv(self.cov)
            
        self.information = self.precision @ self.mean
        self.dim = self.mean.shape[0]

    def marginalize(self, keep_indices):
        keep_indices = np.atleast_1d(keep_indices).astype(int)
        new_mean = self.mean[keep_indices]
        new_cov = self.cov[np.ix_(keep_indices, keep_indices)]
        return Gaussian(new_mean, new_cov)

    def condition(self, observed_indices, observed_values):
        observed_indices = np.atleast_1d(observed_indices).astype(int)
        observed_values = np.atleast_1d(observed_values).astype(float)
        
        all_indices = np.arange(self.dim)
        keep_indices = np.setdiff1d(all_indices, observed_indices)
        
        if len(keep_indices) == 0:
            return None 
            
        mu_A = self.mean[keep_indices]
        mu_B = self.mean[observed_indices]
        
        Sigma_AA = self.cov[np.ix_(keep_indices, keep_indices)]
        Sigma_AB = self.cov[np.ix_(keep_indices, observed_indices)]
        Sigma_BA = self.cov[np.ix_(observed_indices, keep_indices)]
        Sigma_BB = self.cov[np.ix_(observed_indices, observed_indices)]
        
        try:
            inv_Sigma_BB = np.linalg.inv(Sigma_BB)
        except np.linalg.LinAlgError:
            inv_Sigma_BB = np.linalg.pinv(Sigma_BB)
            
        gain = Sigma_AB @ inv_Sigma_BB
        new_mean = mu_A + gain @ (observed_values - mu_B)
        new_cov = Sigma_AA - gain @ Sigma_BA
        
        return Gaussian(new_mean, new_cov)

    def __mul__(self, other):
        if not isinstance(other, Gaussian):
            raise TypeError("Can only multiply Gaussian with another Gaussian.")
        if self.dim != other.dim:
            raise ValueError("Gaussians must have the same dimensionality to be multiplied.")

        new_precision = self.precision + other.precision
        try:
            new_cov = np.linalg.inv(new_precision)
        except np.linalg.LinAlgError:
            new_cov = np.linalg.pinv(new_precision)
            
        new_info = self.information + other.information
        new_mean = new_cov @ new_info
        
        try:
            c = multivariate_normal.pdf(self.mean, mean=other.mean, cov=self.cov + other.cov, allow_singular=True)
        except Exception:
            c = 0.0 

        return Gaussian(new_mean, new_cov), c

    def __repr__(self):
        return f"Gaussian(dim={self.dim}, mean={self.mean})"

class GaussianMixture:
    def __init__(self, weights, components):
        if len(weights) != len(components):
            raise ValueError("Number of weights must match number of components.")
            
        self.weights = np.array(weights).astype(float)
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
            
        self.components = components
        self.dim = components[0].dim if components else 0

    def marginalize(self, keep_indices):
        new_components = [comp.marginalize(keep_indices) for comp in self.components]
        return GaussianMixture(self.weights.copy(), new_components)

    def condition(self, observed_indices, observed_values):
        new_components = []
        new_weights = []
        
        for w, comp in zip(self.weights, self.components):
            mu_B = comp.mean[observed_indices]
            Sigma_BB = comp.cov[np.ix_(observed_indices, observed_indices)]
            
            try:
                likelihood = multivariate_normal.pdf(observed_values, mean=mu_B, cov=Sigma_BB, allow_singular=True)
            except Exception:
                likelihood = 0.0
                
            new_w = w * likelihood
            
            if new_w > 0:
                new_comp = comp.condition(observed_indices, observed_values)
                if new_comp is not None:
                    new_weights.append(new_w)
                    new_components.append(new_comp)
                    
        if sum(new_weights) == 0:
            raise ValueError("Observation has zero probability under all mixture components.")
            
        return GaussianMixture(new_weights, new_components)

    def __mul__(self, other):
        if not isinstance(other, GaussianMixture):
            raise TypeError("Can only multiply GaussianMixture with another GaussianMixture.")
            
        new_weights = []
        new_components = []

        for w1, comp1 in zip(self.weights, self.components):
            for w2, comp2 in zip(other.weights, other.components):
                new_comp, scale = comp1 * comp2
                new_weight = w1 * w2 * scale
                
                if new_weight > 0:
                    new_weights.append(new_weight)
                    new_components.append(new_comp)

        if not new_weights:
            # Fallback to prevent complete collapse when mathematically disjoint
            return GaussianMixture([1.0], [self.components[0]])

        return GaussianMixture(new_weights, new_components)

    def prune(self, threshold=1e-5):
        kept_weights = []
        kept_components = []
        
        # Determine dynamic threshold to prevent total collapse
        max_w = np.max(self.weights)
        dynamic_threshold = min(threshold, max_w / 10.0)
        
        for w, comp in zip(self.weights, self.components):
            if w >= dynamic_threshold:
                kept_weights.append(w)
                kept_components.append(comp)
                
        if not kept_weights:
            # Emergency fallback: keep the highest weighted component
            idx = np.argmax(self.weights)
            kept_weights = [self.weights[idx]]
            kept_components = [self.components[idx]]
            
        self.weights = np.array(kept_weights)
        self.weights /= np.sum(self.weights)
        self.components = kept_components

    def __repr__(self):
        return f"GaussianMixture(components={len(self.components)}, dim={self.dim})"

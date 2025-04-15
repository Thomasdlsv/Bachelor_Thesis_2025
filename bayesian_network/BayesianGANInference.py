import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

class BayesianGANInference:
    def __init__(self, generator, data_sampler_bn, f, sigma_noise):
        """
        Args:
            generator: Generator network (g(z))
            data_sampler_bn: Instance of DataSamplerBN for BN-based sampling
            f: Forward model f(x) connecting latent x to measurement y
            sigma_noise: Covariance matrix Σ of the additive Gaussian noise
        """
        self.generator = generator
        self.data_sampler_bn = data_sampler_bn
        self.f = f
        self.sigma_inv = np.linalg.inv(sigma_noise)

    def _likelihood(self, x, y_hat):
        """p_eta(y_hat - f(x)) where eta is Gaussian noise."""
        fx = self.f(x)
        diff = y_hat - fx
        return np.exp(-0.5 * diff @ self.sigma_inv @ diff.T)

    def monte_carlo_expectation(self, y_hat, l_func, N_samples=1000):
        """Monte Carlo estimate E[l(x)] with BN prior."""
        # Sample latent variables from the Bayesian Network
        z_samples = self.data_sampler_bn.sample_from_bn(N_samples)
        
        # Convert z_samples to a PyTorch tensor
        z_samples = torch.tensor(z_samples, dtype=torch.float32)

        required_dim = self.generator.hidden[0].in_features  # Get the expected input size of the generator
        z_samples = F.linear(z_samples, torch.randn(required_dim, z_samples.shape[1]))

        # Debugging: Print the shape of z_samples
        print("Shape of z_samples:", z_samples.shape)

        # Generate samples using the generator
        x_samples, *_ = self.generator(z_samples)  # Unpack the tuple to get the generated data
        
        # Apply the function l_func to the generated samples
        l_values = l_func(x_samples.detach().numpy())
        
        # Return the mean of l_values
        return np.mean(l_values)

    def mcmc_sampler(self, y_hat, N_samples=1000, step_size=0.1, burn_in=100):
        """Metropolis-Hastings MCMC on z space using BN prior."""
        dim_z = len(self.data_sampler_bn.bn_structure)
        current_z = self.data_sampler_bn.sample_from_bn(1)[0]
        samples = []

        def log_posterior(z):
            x = self.generator(z.reshape(1, -1))[0]
            fx = self.f(x)
            diff = y_hat - fx
            log_likelihood = -0.5 * diff @ self.sigma_inv @ diff.T
            log_prior = -0.5 * np.sum(z**2)
            return log_likelihood + log_prior

        for i in range(N_samples + burn_in):
            proposal = current_z + np.random.normal(scale=step_size, size=dim_z)
            log_accept_ratio = log_posterior(proposal) - log_posterior(current_z)
            if np.log(np.random.rand()) < log_accept_ratio:
                current_z = proposal
            if i >= burn_in:
                samples.append(current_z)

        x_samples = self.generator(np.array(samples))
        return x_samples

    def map_estimate(self, y_hat):
        """Compute z_map using MAP optimization."""
        def r(z):
            z = z.reshape(1, -1)
            x = self.generator(z)[0]
            fx = self.f(x)
            diff = y_hat - fx
            return 0.5 * (diff @ self.sigma_inv @ diff.T + np.sum(z**2))

        dim_z = len(self.data_sampler_bn.bn_structure)
        z0 = self.data_sampler_bn.sample_from_bn(1)[0]
        res = minimize(r, z0, method='L-BFGS-B')
        z_map = res.x
        x_map = self.generator(z_map.reshape(1, -1))[0]
        return x_map

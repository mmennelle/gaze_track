import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Grid size (e.g., 20x20)
grid_size = 20
x = np.linspace(0, 20, grid_size)
y = np.linspace(0, 20, grid_size)
X, Y = np.meshgrid(x, y)

# Define gaze direction mean (mu_g) and joystick direction mean (mu_j)
mu_g = np.array([13, 8])  # gaze vector pointing toward this location
mu_j = np.array([18, 4])  # joystick vector pointing toward this location

# Define covariance matrices (uncertainty)
sigma_g = np.array([[6, 0], [0, 6]])   # gaze uncertainty
sigma_j = np.array([[8, 0], [0, 8]])   # joystick uncertainty

# Define prior P(L) as uniform
prior = np.ones_like(X)

# Gaussian probability density function
def gaussian_2d(X, Y, mu, sigma):
    pos = np.dstack((X, Y))
    inv_sigma = np.linalg.inv(sigma)
    diff = pos - mu
    exponent = np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
    return np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))

# Likelihoods
P_G_given_L = gaussian_2d(X, Y, mu_g, sigma_g)
P_J_given_L = gaussian_2d(X, Y, mu_j, sigma_j)

# Posterior (unnormalized)
posterior = P_G_given_L * P_J_given_L * prior

# Normalize posterior
posterior /= np.sum(posterior)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot posterior heatmap
heatmap = ax.imshow(posterior, extent=(0, 20, 0, 20), origin='lower', cmap='viridis')
ax.grid(True, color='white', linestyle='--', linewidth=0.5)
ax.set_title('Probability of Object Location P(L | G, J)')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
fig.colorbar(heatmap, ax=ax, label='Probability')

# Add gaze and joystick direction points
ax.plot(*mu_g, 'ro', label='Gaze Direction')
ax.plot(*mu_j, 'bo', label='Joystick Direction')
ax.legend()

plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:57:38 2026

@author: Hasan Gökbaş
"""

import numpy as np
import matplotlib.pyplot as plt

# Fixed parameters
r = 1
s = 1

# Parameter ranges
p_vals = np.linspace(-2, 2, 300)
q_vals = np.linspace(-2, 2, 300)

P, Q = np.meshgrid(p_vals, q_vals)

# Trace parameter
X = P*Q*s + P*r + Q*r + s*r

# Eigenvalues of monodromy matrix
alpha = (X + np.sqrt(X**2 + 4*r**3)) / 2
beta  = (X - np.sqrt(X**2 + 4*r**3)) / 2

# Spectral radius of M
rho_M = np.maximum(np.abs(alpha), np.abs(beta))

# Effective spectral radius
rho_eff = rho_M**(1/3)

# Plot
plt.figure(figsize=(10, 8))

heatmap = plt.contourf(
    P, Q,
    rho_eff,
    levels=100,
    cmap='viridis'
)

cbar = plt.colorbar(heatmap)
cbar.set_label(r'$\rho_{\mathrm{eff}}$', fontsize=14)

plt.xlabel('Parameter $p$', fontsize=14)
plt.ylabel('Parameter $q$', fontsize=14)

plt.title(
    r'Heatmap of the Effective Spectral Radius $\rho_{\mathrm{eff}}$'
    '\n'
    r'$(s=1,\ r=1)$',
    fontsize=18
)

plt.tight_layout()
plt.show()

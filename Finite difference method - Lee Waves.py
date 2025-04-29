# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:20:46 2025

@author: lucyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.linalg import solve_banded

U0 = 10
N = 0.03
Lx = 2000
Lz = 2000
nx = 32
nz = 1000
k0 = 2*np.pi/Lx

x = np.linspace(0, Lx, nx, endpoint=False)
z = np.linspace(0, Lz, nz)
dz = z[1] - z[0]

# Boundary 
#h =100*np.sin(k0*x) 
def gaussian_wave(x, peaks, A=300, sigma=130):
     h = np.zeros_like(x)
     for x0 in peaks:
      h += A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
     return h 

# Generate the terrain profile
#peaks = [700,800,900,1000,1100,1200]
peaks = [600,1000,1400]
h = gaussian_wave(x, peaks)


# Compute Fourier transform of h(x)
k_vals = np.zeros(nx)
k_vals[:int(nx/2)] = np.arange(int(nx/2))
k_vals[int(nx/2):] = np.arange(-int(nx/2),0)
k_vals = k_vals*2*np.pi/Lx
H_k = fft(h)/nx
print(k_vals)

# Solve for each mode k
W = np.zeros((nz, nx), dtype=complex)
wk = np.zeros((nz,nx),dtype=complex) 
for i, k in enumerate(k_vals):
    if k == 0:
        continue
    
    # Boundary conditions we worked out
    Z_k0 = U0 * (1j * k) ** 3 * H_k[i]
    
    # Set up finite difference matrix 
    coeff = (N**2 / U0**2) - k**2
    
    # Tridiagonal matrix setup
    A = np.zeros((3, nz), dtype=complex)
    A[0, 2:] = 1  # Upper diagonal
    A[1, 1:-1] = -2 + dz**2 * coeff  # Middle diagonal
    A[2, :-2] = 1  # Lower diagonal
    
    # boundary conditions as b
    b = np.zeros(nz, dtype=complex)
    b[1] = -Z_k0
    #b[1] = Z_kpp0 * dz**2  
    
    # Solve tridiagonal system using built in function
    Z_k = solve_banded((1, 1), A[:, 1:-1], b[1:-1])
    
    W[1:-1, i] = Z_k
    W[0,i] = -Z_k0 
    wk[:,i] = -W[:,i]/(k**2)
    
    

# Inverse Fourier Transform to get W(x, z)
W_real = np.real(ifft(W, axis=1))
wr = np.real(ifft(wk, axis=1))



# Plot results
X, Z = np.meshgrid(x, z)
plt.figure(figsize=(6, 4))
plt.contourf(X, Z,wr, levels=100, cmap='RdYlBu')
plt.plot(x, h, color='black', linewidth=1)  # Plot terrain boundary
plt.colorbar(label='Vertical Velocity w(x,z) ')
plt.title('Numerical Solution for constant U and N')
plt.xlabel('x')
plt.ylabel('z')
plt.show()

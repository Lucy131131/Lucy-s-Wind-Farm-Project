# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:39:52 2025

@author: lucyb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.linalg import solve_banded

U0 = 10
Lx = 2000
Lz = 2000
nx = 32
nz = 1000
H = 5000
h0 = 200
k0 = 2*np.pi/Lx
N0 = 0.055
z0 = 100
Ns = 0.01
alpha = 10**-5

x = np.linspace(0, Lx, nx, endpoint=False)
z = np.linspace(0, Lz, nz)
dz = z[1] - z[0]

Uwind = 7
Uhub = 5
#U0 = Uwind*(Lz**2-(z-Lz)**2)/Lz**2 + Uhub
#U0 = Uwind + Uhub

# Boundary 
#h = h0*np.cos(k0*x) 
def gaussian_wave(x, peaks, A=300, sigma= 140):
     h = np.zeros_like(x)
     for x0 in peaks:
      h += A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
     return h 

# Generate the terrain profile
#peaks = [6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000]
peaks = [600,1000,1400]
h = gaussian_wave(x, peaks)

# Define a step function for N(z)
def N_profile(z):
    z_c = 500 # main SBL top [m]
    
    N_s = 0.02
    N_f = 0.06
    
    N = np.zeros_like(z)
    
    # Apply piecewise definitions
    ABL = (z >= 0) & (z <= z_c)
    N[ABL] = N_s
    N[z > z_c] = N_f
    
    # If input was scalar, return scalar
    return N if len(N) > 1 else N[0]

# Compute Fourier transform of h(x)
k_vals = np.zeros(nx)
k_vals[:int(nx/2)] = np.arange(int(nx/2))
k_vals[int(nx/2):] = np.arange(-int(nx/2),0)
k_vals = k_vals*2*np.pi/Lx
H_k = fft(h)/nx


N_vals = N_profile(z)

# Solve for each mode k
W = np.zeros((nz, nx), dtype=complex)
wk = np.zeros((nz, nx), dtype=complex) 

for i, k in enumerate(k_vals):
    if k == 0:
        continue
    
    # Boundary condition from Fourier transform
    Z_k0 = U0* (1j * k) ** 3 * H_k[i]
    
    # Tridiagonal matrix setup
    A = np.zeros((3, nz), dtype=complex)
    coeff = (N_vals**2 / U0**2) - k**2  
    
    A[0, 2:] = 1  
    A[1, 1:-1] = -2 + dz**2 * coeff[1:-1]  
    A[2, :-2] = 1  
    
    # Boundary conditions in b
    b = np.zeros(nz, dtype=complex)
    b[1] = -Z_k0
 
    
    # Solve tridiagonal system
    Z_k = solve_banded((1, 1), A[:, 1:-1], b[1:-1])
    
    W[1:-1, i] = Z_k
    W[0, i] = -Z_k0 
    wk[:, i] = -W[:, i] / (k**2)


# Inverse Fourier Transform to get W(x, z)
wr = np.real(ifft(wk, axis=1))



# Plot results
X, Z = np.meshgrid(x, z)

wk_k0 = np.zeros_like(wk)
k_pos = np.where(k_vals == k0)[0][0]
k_neg = np.where(k_vals == -k0)[0][0]
wk_k0[:, k_pos] = wk[:, k_pos]
wk_k0[:, k_neg] = wk[:, k_neg]

wr_k0 = np.real(ifft(wk_k0, axis=1))


plt.figure(figsize=(6, 4))
plt.contourf(X, Z, wr, levels=100, cmap='RdYlBu')
plt.colorbar(label='Vertical Velocity')
plt.title('Numerical solution for varying N')
#plt.plot(x, h, color='black', linewidth=1) 
plt.xlabel('x')
plt.ylabel('z')
plt.show()

# Plotting
#plt.figure(figsize=(6, 4))
#plt.plot(N_vals, z)
#plt.xlabel('Brunt–Väisälä Frequency N (s⁻¹)')
#plt.ylabel('Height z (m)')
#plt.title('N(z) profile for a Nocturnal Boundary Layer')
#plt.grid(True)
#plt.tight_layout()
#plt.show()
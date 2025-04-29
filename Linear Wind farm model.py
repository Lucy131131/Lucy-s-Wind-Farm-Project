# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:57:50 2024

@author: lucyb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

Nx = 256
Ny = 256
Lx = 12
Ly = 12
Fr = 0.45
PN = 1
PC = 10
PF = -6.25

dx = Lx/Nx
dy = Ly/Ny

ax=1
bx=2

x = np.linspace(-Lx/2, Lx/2 - dx, Nx)
y = np.linspace(-Ly/2, Ly/2 - dy, Ny)


xv, yv = np.meshgrid(x,y)
plt.plot(xv, yv, marker='o' , linestyle='none')
plt.show()
Fxr = np.zeros((Nx, Ny))

#In Fourier Space
kx = np.zeros(Nx)

kx[:int(Nx/2)] = np.arange(int(Nx/2))

kx[int(Nx/2):] = np.arange(-int(Nx/2),0)

ky = np.zeros(Ny)

ky[:int(Ny/2)] = np.arange(int(Ny/2))

ky[int(Ny/2):] = np.arange(-int(Ny/2),0)

kx = kx/Lx
ky = ky/Ly

kxv, kyv = np.meshgrid(kx, ky)

for i in range(len(xv)-1):
    for j in range(len(yv)-1):
        if abs(xv[i][j]) <= ax/2 and abs(yv[i][j]) <= bx/2:
            Fxr[i][j] = 1

U_B = 1
V_B = 0
V = 0
U = 2

Fxk = np.fft.fft2(Fxr)
sigma_b = U_B*kxv + V_B*kyv
sigma = U*kxv + V*kyv
sb_PC = 1j*sigma_b + PC**(-1)
kx2_ky2 = (kxv**(2) + kyv**(2))**(1/2)

#plt.pcolor(xv, yv, Fxr)

#Displacement equation
denom = (sb_PC*sigma_b) + (PN**(-1)*sigma*kx2_ky2) - (1j*Fr**(-2)*kx2_ky2**2)
etaTk = (-PF**(-1)*kxv *Fxk)/(denom)
etaTk[0][0] = 0
etaTr = np.real(np.fft.ifft2(etaTk))

#plt.pcolor(xv, yv, etaTr, cmap='RdYlBu',vmin = -0.1 , vmax = 0.1)
#plt.colorbar()

#Streamwise Velocity Perturbations
A1 = PN**(-1)*kxv*sigma*etaTk/(sb_PC*kx2_ky2)
A2 = 1j*Fr**(-2)*kxv*etaTk/(sb_PC)
A3 = PF**(-1)*Fxk/(sb_PC)
ubpk = A1 - A2 + A3
ubpk[0][0] = 1
ubpr = np.real(np.fft.ifft2(ubpk))

#plt.pcolor(xv, yv, ubpr, cmap='RdYlBu', vmin = -0.5 , vmax = 0.5)  
#plt.colorbar()

#Spanwise Velocity Perturbation
B1 = (PN**(-1)*kyv*sigma*etaTk)/(sb_PC*kx2_ky2)
B2 = (1j*Fr**(-2)*kyv*etaTk)/(sb_PC)
vbpk = B1 - B2
vbpk[0][0] = 1
vbpr = np.real(np.fft.ifft2(vbpk))

#plt.pcolor(xv, yv, vbpr, cmap='bwr')
#plt.colorbar()

#Pressure Perturbation
pbk = (Fr**(-2) + 1j*PN**(-1)*(sigma/kx2_ky2))*etaTk
pbk[0][0] = 1
pbr = np.real(np.fft.ifft2(pbk))
            
plt.pcolor(xv, yv, pbr, cmap='RdYlBu', vmin=-0.2 , vmax=0.2)
plt.colorbar()

rect=mpatches.Rectangle((-0.5,-1), 1, 2, fill = False, color="black", linewidth = 2)
plt.gca().add_patch(rect)








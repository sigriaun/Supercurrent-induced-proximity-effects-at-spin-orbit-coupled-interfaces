# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 23:39:33 2023

@author: Sigrid Aunsmo 
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:29:33 2023

@author: Sigrid
"""


import numpy as np 
import matplotlib.pyplot as plt
import functions as func 
import save_and_load_data as data 
from matplotlib import cm
import cmath as cmath 
import matplotlib.colors as colors 
import scipy as sp
from tqdm import trange
import time
from matplotlib import rcParams

rcParams['font.size'] = 16
rcParams['figure.figsize'] = (6,4)

def new_Es():
    E1 = np.linspace(0,0.2,15)
    E2 = np.linspace(0.2001,0.9,15)
    E3 = np.linspace(0.9001,1.2,20)
    E4 = np.linspace(1.2001,3,5)
    return np.concatenate((E1,E2,E3,E4))


filename = 'different_triplet_currents'


nphi = 29


phis = np.pi*np.linspace(0,2,nphi)
Es = new_Es()

G_0 = 1
func.G_0 = G_0
"""parameters for material (L)"""
theta = 0*np.pi/4  #angle between the interface magnetizations
P = 0 #polarization 
func.G_phi = G_0*0.3
m = 10
func.m = m

deltaE = 0.001
func.nx = 500
func.L = 8
func.nnodes = 2000 #max nodes for scipy solve_bvp 
func.t = 1e-4 #tolerance for scipy solve_bvp 
func.nm = 8
yv_init = np.zeros((func.nm*2*2,func.nx),dtype = complex)  #initial guess for soulution
func.id2 = np.eye(2, dtype = complex)
func.x = np.linspace(0,func.L,func.nx)
x1 = func.x 
func.Lh = 3
func.G_1  = G_0*( 1 - np.sqrt(1-P**2))/(1 + np.sqrt(1-P**2) )
func.G_MR = G_0* P /(1 + np.sqrt(1- P**2))
func.bc_left = func.bc_spin_active
func.bc_right = func.bc_spin_active
nx1 = func.nx


# func.mL = m*np.array([1,0,0])
# func.h0 = np.array([0,0,150])

mLs = m*np.array([[1,0,0],[0,1,0],[0,0,1]])
h0s = 150*np.array([[0,0,1],[0,0,1],[0,1,0]])

# func.mL = m*np.array([1,0,0])
# #func.mR = m*np.array([np.cos(theta), np.sin(theta), 0])
# func.mR = m*np.array([1,0,0])
# func.h0 = np.array([0,0,150])




"""solve system in the material (L)"""
# tstart = time.time()
# for i in range(3):
#     func.mL = mLs[i]
#     func.mR = mLs[i]
#     func.h0 = h0s[i]
#     #print(func.mL, func.mR, func.h0)
#     for j in trange(len(phis)):
#         phi = phis[j]
#         func.phi = phi
#         for k in range(len(Es)):
#             #print(f'hei, i{i}, j{j}, k{k}')
#             func.E = Es[k] + 1j*deltaE
#             func.b = func.calc_b(func.E)
#             func.tb = np.conj(func.calc_b(-func.E))
#             func.set_up_gmL(func, func.gm_BCS, -phi/2)
#             func.set_up_gmR(func, func.gm_BCS, phi/2)
#             yv_init = func.sovle_single_E_general(func,yv_init,f'{filename}_L_phi{j}_E_{k}_dir{i}', save = 'only_y')
# tend = time.time()
# print(tend-tstart)                          
               
'''Save the middle of (L)'''
# for i in range(3):
#     for j in trange(len(phis)):
#         for k in range(len(Es)):
#             y = np.load(f'data2/{filename}_L_phi{j}_E_{k}_dir{i}_y.npy')
#             ym = func.v_to_m(y)
#             ym_middle = ym[:,:,:,int(func.nx/2)]
#             np.save( f'data2/{filename}_L_phi{j}_E_{k}_dir{i}_middle.npy', ym_middle)
        
"""Parameters for material (R)"""
func.t = 1e-4
func.L = 8
func.nx = 400
func.x = np.linspace(0,func.L,func.nx)
func.pref1 = 0.1
func.pref2 = 0.1
func.pref3 = 0.1
func.omega = 0.1
func.h0 = np.array([0,0,0])
func.bc_left = func.Rashba_bc
func.bc_right = func.bc_KL  
func.set_up_gmR(func,func.gm_vacum)               
yv_init = np.zeros((func.nm*2*2,func.nx),dtype = complex) 
func.alpha = 0 

"""solve system in the material (R)"""
# tstart = time.time()
# for i in range(3): 
#     for j in trange(len(phis)):
#         phi = phis[j]
#         for k in range(len(Es)):
#             func.E = Es[k] + 1j*deltaE
#             func.b = func.calc_b(func.E)
#             func.tb = np.conj(func.calc_b(-func.E))
#             func.set_up_gmL_from_file_alpha(func, f'data2/{filename}_L_phi{j}_E_{k}_dir{i}_middle.npy')
#             yv_init_not = func.sovle_single_E_general(func,yv_init,f'{filename}_R_alpha{i}_phi{j}_E{k}_dir{i}', save = 'only_y')
# tend = time.time()
# print(tend-tstart)               


# M_Idi_dE = np.zeros((3,len(phis),len(Es)))
# for i in range(3): 
#     for j in range(len(phis)):
#         for k in range(len(Es)):
#             y = np.load(f'data2/{filename}_R_alpha{i}_phi{j}_E{k}_dir{i}_y.npy')
#             ym = data.v_to_m(y)
#             M = data.magnetization_one_E(ym)[:,0]
#             M_Idi_dE[i,j,k] = np.linalg.norm(M)
# M_Idi  = sp.integrate.simps(M_Idi_dE,Es, axis = 2)
# np.save(f'data2/{filename}_M_Idi.npy',M_Idi)


M_Idi = np.load(f'data2/{filename}_M_Idi.npy')

plt.figure()
plt.plot(phis, M_Idi[0], label = r'$J_{d_x}$')
plt.plot(phis, M_Idi[1], label = r'$J_{d_y}$')
plt.plot(phis, M_Idi[2], '--',label = r'$J_{d_z}$')
plt.legend(loc = 'upper right')
plt.ylabel(r'$M/M_0$')
plt.xlabel(r'$\phi$')
plt.xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],['0', r'$\pi/2$','$\pi$', '$3\pi/2$', r'$2\pi$'])
plt.savefig("plots/triplet magnetization Jdx Jdy Jdz.pdf", bbox_inches="tight", format="pdf")
plt.show()            

            




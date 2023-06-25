# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:55:49 2023

@author: Sigrid Aunsmo 
"""

import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange

import functions as func 
import save_and_load_data as data 

# def new_Es():
#     E1 = np.linspace(0.0001,0.2,15)
#     E2 = np.linspace(0.2001,0.9,15)
#     E3 = np.linspace(0.9001,1.2,20)
#     E4 = np.linspace(1.2001,3,5)
#     return np.concatenate((E1,E2,E3,E4))
    
def new_Es():
    #E1 = np.linspace(0.001,0.2,15)
    #E2 = np.linspace(0.2001,0.9,15)
    #E3 = np.linspace(0.9001,1.2,20)
    #E4 = np.linspace(1.2001,3,5)
    #return np.concatenate((E1,E2,E3,E4))
    #E1 = np.linspace(0.001,0.99,100)
    #E2 = np.linspace(1.001,3,20)
    #return np.concatenate((E1,E2))
    #return np.linspace(0.0001,2,301)
    #return np.concatenate((np.linspace(0.0001,1.5,101), np.linspace(1.6,3,20)))
    return np.concatenate((np.linspace(0.0001,0.07,50),np.linspace(0.08,1.5,90), np.linspace(1.6,3,20)))


Es = new_Es()




#filename = 'temperature_dependecy_of_mx_2' #L = 8 #
#filename = 'temperature_dependecy_of_mx_3' #L = 1
#filename = 'temperature_dependecy_of_mx_4' #L = 2
filename = 'temperature_dependecy_of_mx_5' #L = 1, 
filename = 'temperature_dependecy_of_mx_6' 
filename = 'temperature_dependecy_of_mx_7' 
filename = 'temperature_dependecy_of_mx_8' 
filename = 'temperature_dependecy_of_mx_9' 
filename = 'mx_of_T' #did not seem to be good enough... 
filename = 'mx_of_T_2'

func.deltaE = 0.01 #1e-6   ##1e-3
func.nx = 500
func.L = 8
func.nnodes = 1000 #max nodes for scipy solve_bvp 
func.t = 1e-4 #tolerance for scipy solve_bvp 
func.nm = 8
yv_init = np.zeros((32,func.nx), dtype = complex)
func.id2 = np.eye(2, dtype = complex)
func.x = np.linspace(0,func.L,func.nx) # Es = np.concatenate(np.linspace(0.0001,1.5,351), np.linpace(1.6,3,10))
func.Lh = 3

#beta_prime = 20

#Ts = np.linspace(0.01,0.99,30)
Ts = np.linspace(0.01,0.99,40)
betas_prime = 1/Ts

'''Josepshoson junction parameters'''
func.h0 = np.array([0,0,0])
func.bc_left = func.bc_KL
func.bc_right = func.bc_KL
func.omega = 0.01
phi = np.pi/4

'''solving for material L'''
# for t in trange(len(betas_prime)):
#     beta_prime = betas_prime[t]
#     for j in range(len(Es)):
#         func.E0 = Es[j]
#         func.E = func.E0 + 1j*func.deltaE
#         func.b = func.calc_b(func.E/np.tanh(1.74*np.sqrt( beta_prime - 1 )))
#         func.tb = -func.b        
#         func.set_up_gmL(func,func.gm_BCS, phi/2)
#         func.set_up_gmR(func,func.gm_BCS, -phi/2)
#         yv_init = np.zeros((32,func.nx), dtype = complex)
#         func.sovle_single_E_general(func,yv_init,f'{filename}_first_E{j}_beta{t}', save = 'only_y')

# for t in range(len(betas_prime)):
#     for j in range(len(Es)):
#         E = Es[j]
#         y = np.load(f'data2/{filename}_first_E{j}_beta{t}_y.npy')
#         middle_index_1 = int(len(y[0])/2)  
#         ym = data.v_to_m(y)
#         np.save(f'data2/{filename}_middle_E{j}_beta{t}.npy',ym[:,:,:,middle_index_1])    
 
'''Rashba interface parameters'''         
func.h0 = np.array([0,0,0])
func.pref1 = 0.1
func.pref2 = 0.1
func.pref3 = 0.1
func.omega = 0.1
func.bc_left = func.Rashba_bc
func.bc_right = func.bc_KL
func.L = 8
func.x = np.linspace(0,func.L,func.nx)

'''solving for material R'''
# for t in trange(len(betas_prime)):    
#     beta_prime = betas_prime[t]
#     for j in range(len(Es)):
#         E = Es[j]
#         func.E0 = E
#         func.E = E + 1j*func.deltaE
#         func.b = func.calc_b(func.E/np.tanh(1.74*np.sqrt( beta_prime - 1 )))
#         func.tb = - func.b
#         func.set_up_gmL_from_file(func,f'data2/{filename}_middle_E{j}_beta{t}.npy')
#         func.set_up_gmR(func,func.gm_vacum)
#         func.sovle_single_E_general(func,yv_init,f'{filename}_sencond_E{j}_beta{t}', save = 'only_y')


'''plotting magnetization'''
# dmxL = np.zeros((len(Es),len(betas_prime),func.nx))
# mxL = np.zeros((len(betas_prime),func.nx))
# for t in trange(len(betas_prime)):
#     beta_prime = betas_prime[t]
#     for j in range(len(Es)): 
#         y = np.load(f'data2/{filename}_sencond_E{j}_beta{t}_y.npy')
#         ym = data.v_to_m(y)
#         fsi, dxi, dyi, dzi, tfsi, tdxi, tdyi, tdzi  = data.calc_f_from_gamma(ym)
#         mx = np.real(fsi*tdxi - tfsi*dxi) 
#         dmxL[j,t] = mx*np.tanh(1.76*beta_prime*Es[j]/2)
#     mxL[t,:] = sp.integrate.simps(dmxL[:,t,:], Es, axis = 0) 
# np.save(f'data2/{filename}_mx_of_T.npy',mxL)
'''Loading'''
mxL = np.load(f'data2/{filename}_mx_of_T.npy')

plt.figure()
#plt.title('mx')
plt.plot(1/betas_prime[:-2],-mxL[:-2,-1])
plt.hlines(0,0,1)
plt.yscale("log")
plt.ylabel(r'$M_x/M_0$')
plt.xlabel(r'$T/T_c$')
plt.savefig("plots/singlet new mx of T.pdf", bbox_inches="tight", format="pdf")
plt.savefig("plots/singlet new mx of T.svg", bbox_inches="tight", format="svg")
plt.show()

'''plotting current'''
# dJL = np.zeros((len(Es),len(betas_prime)))
# JL = np.zeros(len(betas_prime))
# for t in range(len(betas_prime)):
#     beta_prime = betas_prime[t]
#     for j in range(len(Es)): 
#         y = np.load(f'data2/{filename}_first_E{j}_beta{t}_y.npy')
#         ym = data.v_to_m(y)
#         J, Jsx, Jsy, Jsz, I_simp, ISx_simp, ISy_simp, ISz_simp,dfs,ddx,ddy,ddz,dtfs,dtdx,dtdy,dtdz,I_fs, I_dx, I_dy, I_dz = data.currents(ym)
#         dJL[j,t] = J[int(func.nx/2)]
#     JL[t] = sp.integrate.simps(dJL[:,t]*np.tanh(1.76*beta_prime*Es/2), Es, axis = 0) 
# np.save(f'data2/{filename}_J_of_T.npy',JL)
'''loading'''
JL = np.load(f'data2/{filename}_J_of_T.npy')

plt.figure(figsize = (6,6))
plt.title('current')
plt.plot(1/betas_prime[0:],JL[0:])
plt.hlines(0,0,1)
plt.show()


'''plot magnetization as a function of T and y'''
X,Y = np.meshgrid(Ts,func.x)
Z = mxL 
Zmax = np.max(np.abs(mxL))
plt.figure(figsize = (6,6))
plt.title(f'mx at phi {phi}')
plt.imshow(Z.T,vmax = Zmax, vmin = -Zmax, cmap = cm.RdBu,aspect='auto', interpolation='nearest')
plt.colorbar()
plt.show()

plt.plot(func.x, mxL[2])
plt.show()

'''plot integrand'''
X, Y = np.meshgrid(func.x,Es)
Z = dmxL[:,0,:]
Zmax = np.max(np.abs(Z))
plt.pcolormesh(X,Y,Z,vmax = Zmax, vmin = -Zmax, cmap = cm.RdBu)
plt.colorbar()
plt.show()

# plt.plot(Es,dmxL[:,0,0])
# '''plot mx as a function of y for one T'''






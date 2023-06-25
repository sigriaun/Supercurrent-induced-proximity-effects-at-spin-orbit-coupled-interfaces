# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:07:36 2023

@author: Sigrid Aunsmo"""
import numpy as np
from scipy.integrate import solve_bvp 
import matplotlib.pyplot as plt
import time
from numba import jit
import numba


import save_and_load_data as data
#from save_and_load_data import *

sigmax = np.array([[0,1],[1,0]],dtype = complex)  
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]],dtype = complex)
sigma = np.array([sigmax,sigmay,sigmaz])
#rhotre = np.diag(1,1,-1,-1, dtype = complex )
id2 = np.eye(2,dtype = complex)


#@jit(nopython = True)
def v_to_m(v): 
    #global nm
    n1,n2 = np.shape(v)
    nx = n2
    nm = int(n1/4)
    return np.reshape(v,(nm,2,2,nx))

# @jit(nopython = True)
# def m_to_v(m): 
#     #global nm 
#     n1,n2,n3,n4 = np.shape(m)
#     nx = n4
#     nm = n1
#     return np.reshape(m,(nm*2*2,nx))

#@jit(nopython = True)
def m_to_v(m,nx): 
    #global nm 
    #n1,n2,n3,n4 = np.shape(m)
    #nx = n4
    #nm = n1
    #return np.reshape(m,(8*2*2,nx))
    m = m.reshape((32,nx))
    return m 

def calc_b(Ei): 
    if np.abs(np.real(Ei)) < 1: 
        #print(Ei)
        #print('b: 1')
        return 1/( Ei + 1j*np.sqrt(1 - Ei**2) )
    else:
        #print('b: 2')
        #return np.sign(E)/( np.sqrt(E**2 - 1) + E*np.sign(E) )
        return np.sign(Ei)/( np.sqrt(Ei**2 - 1) + np.abs(Ei) )
    
#@jit(nopython = True)
def create_varying_h(x, nxlocal):
   '''
   need global variable:
   -Lh  
   -L
   -h0
   ''' 
   #print('h0' ,h0)
   h = np.zeros((3,len(x)))
   h[0] = h0[0]*(1 - np.exp(-(L/2 - x)**2/Lh))
   h[1] = h0[1]*(1 - np.exp(-(L/2 - x)**2/Lh))
   h[2] = h0[2]*(1 - np.exp(-(L/2 - x)**2/Lh))
   #h[x<Lx] = h0
   #h[x>L-Lx] = h0
   
   '''quick fix to get constant h'''
   #h[0,:] = h0[0]
   #h[1,:] = h0[1]
   #h[2,:] = h0[2]
   #id2 = np.array([[1+ 0j,0],[0,1 +0j]])
   #id2nx = np.zeros((2,2, nxlocal),dtype=numba.complex64)
    
   return h

#@jit(nopython = True)
def fun(x,y): 
    ym = v_to_m(y)
    #ym = np.reshape(y,(nm,2,2))
    
    gm  = ym[0]  + 1j*ym[4]
    et  = ym[1]  + 1j*ym[5]
    tgm = ym[2]  + 1j*ym[6]
    tet = ym[3]  + 1j*ym[7]  
    
    #dym = np.zeros_like(ym)
    
    #id2 = np.eye(2,dtype=complex)
    id2 = np.array([[1+ 0j,0],[0,1 +0j]])
    
    nxlocal = len(gm[0,0])
    N = np.zeros_like(gm)
    tN = np.zeros_like(gm)
    
    dym = np.zeros((nm,2,2,nxlocal))
    
    #id2nx = np.zeros((2,2, nxlocal), dtype = complex)
    id2nx = np.zeros((2,2, nxlocal),dtype=complex)
    id2nx[0,0] = 1
    id2nx[1,1] = 1
       
    M = id2nx - np.einsum('ijk,jlk->ilk', gm, tgm)
    Mdet = M[0,0]*M[1,1]-M[0,1]*M[1,0] 
    N[0,0] = 1/Mdet * M[1,1]
    N[1,1] = 1/Mdet * M[0,0]
    N[1,0] = -1/Mdet * M[1,0]
    N[0,1] = -1/Mdet * M[0,1]
    
    tM = id2nx - np.einsum('ijk,jlk->ilk', tgm, gm)
    tMdet = tM[0,0]*tM[1,1]-tM[0,1]*tM[1,0] 
    tN[0,0] = 1/tMdet * tM[1,1]
    tN[1,1] = 1/tMdet * tM[0,0]
    tN[1,0] = -1/tMdet * tM[1,0]
    tN[0,1] = -1/tMdet * tM[0,1]
    
    # for i in range(nxlocal):
    #     N[:,:,i]  = np.linalg.inv( id2 -  gm[:,:,i] @ tgm[:,:,i] ) 
    #     tN[:,:,i] = np.linalg.inv( id2 - tgm[:,:,i] @  gm[:,:,i] )
    
    sigmagamma  = np.einsum('ijk,klm->ijlm', sigma, gm) - np.einsum('jkm,ikl-> ijlm',gm,np.conj(sigma))
    tsigmagamma = np.einsum('ijk,klm->ijlm', np.conj(sigma), tgm) - np.einsum('jkm,ikl-> ijlm',tgm,sigma)
    '''create h'''
    h = create_varying_h(x,nxlocal)
    
    
    # print('nxlocal', nxlocal)
    # print(np.shape(h))
    # print(np.shape(sigmagamma))
    #test = 1j*np.einsum('il,ijkl->jkl',h, sigmagamma  ) 
    
    
    '''d gamma/dz'''
    #print(E0)
    dgm  = et
    det = - 2j*E*gm  - 1j*np.einsum('il,ijkl->jkl',h, sigmagamma  )   - 2* np.einsum('ijk,jlk->ilk', et  , np.einsum('ijk,jlk->ilk', tN , np.einsum('ijk,jlk->ilk', tgm , et  )))
    #det = - 2j*E*gm   - 2* np.einsum('ijk,jlk->ilk', et  , np.einsum('ijk,jlk->ilk', tN , np.einsum('ijk,jlk->ilk', tgm , et  )))
    dtgm  = tet
    dtet = - 2j*E*tgm + 1j*np.einsum('il,ijkl->jkl',h, tsigmagamma )  - 2* np.einsum('ijk,jlk->ilk', tet , np.einsum('ijk,jlk->ilk', N  , np.einsum('ijk,jlk->ilk',  gm , tet )))
    #dtet = - 2j*E*tgm  - 2* np.einsum('ijk,jlk->ilk', tet , np.einsum('ijk,jlk->ilk', N  , np.einsum('ijk,jlk->ilk',  gm , tet )))
    '''Setiing up the return matrix'''
    dym[0] = np.real(dgm)
    dym[1] = np.real(det)
    dym[2] = np.real(dtgm)
    dym[3] = np.real(dtet)
    dym[4] = np.imag(dgm)
    dym[5] = np.imag(det)
    dym[6] = np.imag(dtgm)
    dym[7] = np.imag(dtet)
    yv = m_to_v(dym, nxlocal)

    return yv

 
def bc_spin_active(tilde,LR,gm,tgm,et,tet,Na,tNa,gmLR,tgmLR,etLR,tetLR,NLR,tNLR):
    #LR = - LR
    '''
    LR = 1 when evaluating del gamma_L
    LR = -1 when evaluating del gamma_R
    tilde = 1 when evaluating del tgamma
    tilde = -1 when ecaluating del gamma
    This tilde variable should be multiplied with every j that occure in the equation and switch mgamma and mgammac
    
    When evaluating the tilde conjugated equation, the gm and tgm should switch places as inputs
    
    what has to be defined outside solve_bvp: 
        -m_R
        -m_L
        -G_0
        -G_1
        -G_MR
        -G_phi
    '''
    
    '''fixing the coorrect msig and msigc for right/left and tilde-conjucated/non-tildde-conjugated'''
    
    #print(sigma)
    
    if LR == 1:
        m_bc = np.copy(mL)
        #print('L')
    else: 
        m_bc = np.copy(mR)
        #print('R')
    if tilde == -1:
        #print('tilde')
        msig = np.einsum('i,ijk->jk',m_bc,np.conj(sigma))   
        msigc = np.einsum('i,ijk->jk',m_bc,sigma)
       
    else:
        #print('not tilde')
        msig = np.einsum('i,ijk->jk',m_bc,sigma)
        msigc = np.einsum('i,ijk->jk',m_bc,np.conj(sigma))
    
    
    # print('m_bc = ', m_bc)
    # print('msig = ', msig)
    # print('msigc = ', msigc)
    #print('\n msig',msig)
  
    #print('tilde = ', tilde, '\n msig=', msig, '\n msigc= ', msigc)
    '''Creating the N matrices'''
    id2 = np.eye(2, dtype = complex)
    NLR = np.linalg.inv(id2 - gmLR @ tgmLR )
    tNLR = np.linalg.inv(id2 - tgmLR @ gmLR)
    N = np.linalg.inv(id2 - gm @ tgm)
    tN = np.linalg.inv(id2 - tgm @ gm)
    
    
    '''calculating the terms'''
    term_KL   =  (id2 - gm @ tgmLR ) @ NLR @ ( gmLR - gm)
    term_1    =  msig @ NLR @ gmLR @ msigc      -     gm @ msigc @ tNLR @ msigc      +    m**2 * gm     - msig @ NLR @ msig @ gm      +    gm @ msigc @ tNLR @ tgmLR @ msig @ gm     
    term_MR   = (NLR @ gmLR @ msigc +  msig @ NLR @ gmLR )    - gm @ ( tNLR @ msigc  + msigc @ tNLR - msigc )     -   (NLR @ msig - msig + msig @ NLR ) @ gm     + gm @ ( tNLR @ tgmLR @ msig  + msigc @ tNLR @ tgmLR  ) @ gm
    term_phi  = gm @ msigc - msig @ gm 

    '''Including all faactors in the terms . 
    NB! these has to be defined outside of the solve_bvp() function
    '''
    delzgamma = -LR*(G_0 *term_KL + G_1*term_1 + G_MR*term_MR - (1j*tilde) *1/2*G_phi*term_phi )
    #return  -LR*(- (1j*tilde) *1/2*G_phi*term_phi + G_0 *term_KL ) #+ G_MR*term_MR )# + G_1*term_1 )
    return delzgamma 
 
def Rashba_bc(tilde,LR,gma,tgma,eta,teta,Na,tNa,gmLR,tgmLR,etLR,tetLR,NLR,tNLR):
    '''
    Needs global variables 
    sigmax
    sigmay
    pref0
    pref1
    pref2
    pref3
    '''
    eta_Rashba =  bc_KL(tilde,1,gma,tgma,eta,teta,Na,tNa,gmLR,tgmLR,etLR,tetLR,NLR,tNLR)
    eta_Rashba += -pref1*(Rashba_term_1(-sigmax,tilde,gmLR,tgmLR,NLR,tNLR,gma,tgma) + Rashba_term_1(sigmaz,tilde,gmLR,tgmLR,NLR,tNLR,gma,tgma) ) 
    eta_Rashba += pref2*Rashba_term_2(-sigmax,tilde,gmLR,tgmLR,NLR,tNLR,etLR,tetLR,gma,tgma)
    eta_Rashba += pref3*(Rashba_term_3_4(-sigmax,tilde,gmLR,tNLR,NLR,tgmLR,gma,tgma,Na,tNa) + Rashba_term_3_4(sigmaz,tilde,gmLR,tNLR,NLR,tgmLR,gma,tgma,Na,tNa) ) 
    return LR*eta_Rashba

def bc_KL(tilde,LR,gma,tgma,eta,teta,Na,tNa,gmLR,tgmLR,etLR,tetLR,NLR,tNLR):
    #print('LR', LR)
    '''9
    needs global parameter omega
    '''
    return LR*omega * ( id2 - gma @ tgmLR ) @ NLR  @ ( gma - gmLR ) 


def Rashba_term_1(sigma,tilde,gm1,tgm1,N1,tN1,gma,tgma):
    if tilde == 1:
        sigmai = sigma
        sigmai_c = np.conj(sigma)
    elif tilde == -1:
        sigmai = np.conj(sigma)
        sigmai_c = sigma
    else: print('Wrong tilde value in rashba term 1')
    return - sigmai @ N1 @ gm1 @ sigmai_c - gma @ sigmai_c @ tN1 @ sigmai_c + gma - sigmai @ N1 @ sigmai @ gma - gma @ sigmai_c @ tN1 @ tgm1 @ sigmai @ gma

def Rashba_term_2(sigma,tilde,gm1,tgm1,N1,tN1,et1,tet1,gma,tgma): 
    sigmax = sigma 
    sigmax_c = sigma 
    term_2A  =   sigmax @  N1 @ (et1 - gm1 @ tet1 @ gm1 ) @ tN1                 - N1 @ (et1 -  gm1 @ tet1 @ gm1 ) @ tN1 @ sigmax_c   
    term_2B  = - gma @ sigmax_c @ tN1 @ ( tet1 @ gm1   -  tgm1 @ et1 ) @ tN1    - gma @ tN1 @ ( tet1 @ gm1   -  tgm1 @ et1 ) @ tN1 @ sigmax_c
    term_2C  = - sigmax @ N1 @ ( et1 @ tgm1 - gm1 @ tet1 ) @ N1 @ gma           - N1 @ ( et1 @ tgm1 - gm1 @ tet1 ) @ N1 @ sigmax @ gma
    term_2D  =   gma @ sigmax_c @ tN1 @ ( tet1 - tgm1 @ et1 @ tgm1 ) @ N1 @ gma - gma @ tN1 @ ( tet1 - tgm1 @ et1 @ tgm1 ) @ N1 @ sigmax @ gma 
    return term_2A + term_2B + term_2C + term_2D

def Rashba_term_3_4(sigma,tilde,gm1,tN1,N1,tgm1,gma,tgma,Na,tNa):
    if tilde == 1:
        sigmai = sigma
        sigmai_c = np.conj(sigma)
    elif tilde == -1:
        sigmai = np.conj(sigma)
        sigmai_c = sigma
    return -  gma + sigmai @ Na @ gma @ sigmai_c +  gma @ sigmai_c @ tNa @ sigmai_c + sigmai @ Na @ sigmai @ gma + gma @ sigmai_c @ tNa @ tgma @ sigmai @ gma

def bc_vacum(tilde,LR,gma,tgma,eta,teta,Na,tNa,gmLR,tgmLR,etLR,tetLR,NLR,tNLR):
    return np.zeros((2,2),dtype = complex)
 
def sovle_single_E_general(obj,yv_init,filename = None, save = 'all'):
    #print(np.shpape())
    tstart = time.time()
    sol = solve_bvp(fun, bc_general, x , yv_init, max_nodes=nnodes, tol = t)
    tend = time.time()
    time_used =  tend - tstart
    #print('h0' ,h0)
    
    if sol.success != True: 
        print(sol.message)
        print(f'\n E = {E}, phi = {phi} ')
    if filename == None: 
        return sol, time_used
    else: 
        if save == 'all':
            data.save_single_E(sol, E, x, filename)
        elif save == 'middle':
            y  = sol.sol(x)
            ym = v_to_m(y)
            middle_index = int(len(ym[0,0,0])/2)
            ym_middle = ym[:,:,:,middle_index]
            gm_middle  = ym_middle[0]  + 1j*ym_middle[4]
            et_middle  = ym_middle[1]  + 1j*ym_middle[5]
            tgm_middle = ym_middle[2]  + 1j*ym_middle[6]
            tet_middle = ym_middle[3]  + 1j*ym_middle[7] 
            #print(middle_index)
            np.save(f'data2/{filename}_middle.npy', [gm_middle, tgm_middle, et_middle, tet_middle])
            return y 
        elif save == 'mxmy': 
            data.save_mxmy(filename,sol,x)
            y = sol.sol(x)
            return y
        elif save == 'mxmy_fsdxdy': 
            data.save_mxmy_fsdxdy(filename,sol,x)
            y = sol.sol(x)
            return y
        elif save == 'convergence':
            x_con = np.linspace(0,L,100)
            return sol.sol(x_con)
        elif save == 'only_y': 
            np.save(f'data2/{filename}_y.npy',sol.sol(x))
            return sol.sol(x)
        
def solve_multiple_E_general(obj,Es,filename,phase1 = None,phase2 = None):
    #np.zeros((len(Es),8,2,2,len(x_init)))
    yv_init = np.zeros((8*2*2,nx))
    times = time.time()
    for i in range(len(Es)): 
        obj.E = Es[i] + deltaE
        obj.b = calc_b(E)
        obj.tb = np.conj(calc_b(-E))
        set_up_gmL(obj,gmleft_type,phase1)
        set_up_gmR(obj,gmright_type,phase2)
        sol, time_used = sovle_single_E_general(obj,yv_init)
        data.save_single_E(sol,E,x,filename+str(i)+'.npz') 
        yv_init = sol.sol(x) #using the new solution as an inital guess for the new E
    timee = time.time()
    print(f'Time used: {timee-times}')
    data.save_multiple_E_new(filename, Es, x)
    
    
       
def bc_general(ya, yb):
    '''
    Global variables that has to be defined
    id2
    nm
    NE
    gmL
    tgmL
    etL 
    tetL
    NL
    tNL
    gmR
    tgmR
    etR
    tetR
    NR
    tNR
    '''
   
    '''getting gamma, eta and tildeconjugated'''
    yma = np.reshape(ya,(nm,2,2))
    ymb = np.reshape(yb,(nm,2,2))
    gma = yma[0] + 1j*yma[4]
    eta = yma[1] + 1j*yma[5]
    tgma= yma[2] + 1j*yma[6]
    teta= yma[3] + 1j*yma[7]
    gmb = ymb[0] + 1j*ymb[4]
    etb = ymb[1] + 1j*ymb[5]
    tgmb= ymb[2] + 1j*ymb[6]
    tetb= ymb[3] + 1j*ymb[7]
    Na  = np.linalg.inv(id2 -  gma@tgma)
    tNa = np.linalg.inv(id2 - tgma@ gma)
    Nb  = np.linalg.inv(id2 -  gmb@tgmb)
    tNb = np.linalg.inv(id2 - tgmb@ gmb)
    
    '''Find the bounadary conditions'''
    etL_bc    =  bc_left( 1, 1, gma,tgma, eta,teta, Na,tNa, gmL,tgmL, etL,tetL, NL,tNL)
    tetL_bc   =  bc_left(-1, 1,tgma, gma,teta, eta,tNa, Na,tgmL, gmL,tetL, etL,tNL, NL)
    etR_bc    = bc_right( 1,-1, gmb,tgmb, etb,tetb, Nb,tNb, gmR,tgmR, etR,tetR, NR,tNR)
    tetR_bc   = bc_right(-1,-1,tgmb, gmb,tetb, etb,tNb, Nb,tgmR, gmR,tetR, etR,tNR, NR)
    
    # eta_bc = omega*(id2 - gma @ tgmL) @ NL @ (gma - gmL)
    # teta_bc = omega*(id2 - tgma @ gmL) @ tNL @ (tgma - tgmL)
    # etb_bc = omega*(id2 - gmb @ tgmR ) @ NR @ (gmR - gmb)
    # tetb_bc = omega*(id2 - tgmb @ gmR ) @ tNR @ (tgmR - tgmb)
    # return np.reshape( [np.imag(eta-eta_bc),np.real(eta-eta_bc),np.imag(teta-teta_bc),np.real(teta-teta_bc),np.imag(etb-etb_bc), np.real(etb-etb_bc), np.imag(tetb-tetb_bc), np.real(tetb-tetb_bc)], 32)
    
    return  np.reshape([np.imag(etb - etR_bc),np.real(etb - etR_bc),np.imag(tetb - tetR_bc),np.real(tetb - tetR_bc),np.real(eta - etL_bc), np.imag(eta - etL_bc),np.real(teta - tetL_bc),np.imag(teta - tetL_bc)], nm*2*2 )

   
# def gm_bulk_superconductor(E, phi = 0):
#     b  = calc_b(E)
#     tb = np.conj(calc_b(-E))    
#     gm = np.array([[0,b],[-b,0]])*np.exp(1j*phi)
#     tgm = np.array([[0,tb],[-tb,0]])*np.exp(-1j*phi)    
#     return gm,tgm

# def gm_Isz():
#     gmL  = np.array( [[a * np.exp(1j*J) , 0 ],[0 , a * np.exp(-1j*J) ]]  ) 
#     tgmL = np.array( [[ta * np.exp(-1j*J) , 0 ],[0 , ta * np.exp(1j*J) ]]  ) 
#     etL  = 1j*J* np.array( [[a * np.exp(1j*J) , 0 ],[0 , - a * np.exp(-1j*J) ]]  ) 
#     tetL = 1j*J* np.array( [[- ta * np.exp(-1j*J) , 0 ],[0 , ta * np.exp(1j*J) ]]  )
#     return gmL, tgmL, etL, tetL

# def gm_Isy():
#     gm1 = np.array( [[ 1j*a * np.cos(J), - 1j*a* np.sin(J) ],[-1j*a *np.sin(J) , -1j*a*np.cos(J) ]] )
#     tgm1 = np.array( [[ -1j*ta * np.cos(J),  1j*ta* np.sin(J) ],[1j*ta *np.sin(J) , 1j*ta*np.cos(J) ]] )
#     et1 = J*np.array( [[ -1j*a * np.sin(J), - 1j*a* np.cos(J) ],[-1j*a *np.cos(J) , 1j*a*np.sin(J) ]] )
#     tet1 = J*np.array( [[ 1j*ta * np.sin(J), 1j*ta* np.cos(J) ],[1j*ta *np.cos(J) , -1j*ta*np.sin(J) ]] )
#     return gm1, tgm1, etL, tetL

# def gm_Isx():
#     gm1  = np.array( [[a*np.sin(J),-1j*a*np.cos(J)],[-1j*a*np.cos(J),a*np.sin(J)]] )
#     tgm1 = np.array( [[ta*np.sin(J),1j*ta*np.cos(J)],[1j*ta*np.cos(J),ta*np.sin(J)]] )
#     et1 = J*np.array( [[a*np.cos(J),+1j*a*np.sin(J)],[1j*a*np.sin(J),a*np.cos(J)]] )
#     tet1 = J*np.array( [[ta*np.cos(J),-1j*ta*np.sin(J)],[-1j*ta*np.sin(J),ta*np.cos(J)]] )
    
def gm_vacum(): 
    gm1 = tgm1 = et1 = tet1 = np.zeros((2,2),dtype = complex)
    return gm1, tgm1, et1, tet1

# def gm_dx_chargecurrent():
#     a = 0.1 
#     gm1 = np.array( [[a * np.exp(1j*J) , 0 ],[0 ,- a * np.exp(1j*J) ]]  )
#     tgm1 = np.array( [[a * np.exp(-1j*J) , 0 ],[0 ,- a * np.exp(-1j*J) ]]  ) 
#     et1  =  1j*J* np.array( [[a * np.exp(1j*J) , 0 ],[0 , -a * np.exp(1j*J) ]]  ) 
#     tet1 = -1j*J* np.array( [[a * np.exp(-1j*J) , 0 ],[0 , -a * np.exp(-1j*J) ]]  )
#     return gm1, tgm1, et1, tet1

# def gm_dy_chargecurrent():
#     a = 0.1
#     gm1 += np.array( [[1j * a * np.exp(1j*J) , 0 ],[0 , 1j* a * np.exp(1j*J) ]]  )
#     tgm1 += np.array( [[- 1j* a * np.exp(-1j*J) , 0 ],[0 , - 1j*a * np.exp(-1j*J) ]]  ) 
#     et1  +=  1j*J* np.array( [[1j*a* np.exp(1j*J) , 0 ],[0 , 1j*a * np.exp(1j*J) ]]  ) 
#     tet1 += -1j*J* np.array( [[- 1j* a * np.exp(-1j*J) , 0 ],[0 , - 1j* a * np.exp(-1j*J) ]]  )   
#     return gm1, tgm1, et1, tet1
    
# def gm_dz_chargecurrent():
#     a = 0.1
#     gm1 += np.array([[0, a *np.exp( 1j* J)],[a *np.exp( 1j* J),0 ]])
#     tgm1 += np.array([[0, a *np.exp( -1j* J)],[a *np.exp( -1j* J),0 ]])
#     et1 += 1j*J *np.array([[0, a *np.exp( 1j* J)],[a *np.exp( 1j* J),0 ]])
#     tet1 += -1j*J*np.array([[0, a *np.exp( -1j* J)],[a *np.exp( -1j* J),0 ]])
#     return gm1, tgm1, et1, tet1

def gm_BCS(): 
    et1 = tet1 = np.zeros((2,2), dtype = complex)
    gm1 = np.array([[0,b],[-b,0]])
    tgm1 = np.array([[0,tb],[-tb,0]])
    return gm1, tgm1, et1, tet1

# def gm_dx(): 
#     et1 = tet1 = np.zeros((2,2), dtype = complex)
#     gm1, tgm1 = np.array([[-b,0],[0,b]])
#     return gm1, tgm1, et1, tet1
    
def set_up_gmL_from_file(obj,filename,phase = None):
    ym = np.load(filename)
    obj.gmL  = ym[0] + 1j*ym[4]
    obj.etL  = ym[1] + 1j*ym[5]
    obj.tgmL = ym[2] + 1j*ym[6]
    obj.tetL = ym[3] + 1j*ym[7]
    
    # gma = yma[0] + 1j*yma[4]
    # eta = yma[1] + 1j*yma[5]
    # tgma= yma[2] + 1j*yma[6]
    # teta= yma[3] + 1j*yma[7]
    obj.NL = np.linalg.inv(id2 -   obj.gmL @ obj.tgmL)
    obj.tNL = np.linalg.inv(id2 - obj.tgmL @  obj.gmL)
       
def set_up_gmL(obj,gm_type, phase = None): 
    obj.gmL, obj.tgmL, obj.etL, obj.tetL, = gm_type()
    obj.NL = np.linalg.inv(id2 - gmL @ tgmL)
    obj.tNL = np.linalg.inv(id2 - tgmL @ gmL)
    if phase != None:
        #print('\n hei')
        obj.gmL = obj.gmL*np.exp(1j*phase)
        obj.tgmL = obj.tgmL*np.exp(-1j*phase)
    
    
def set_up_gmR(obj,gm_type, phase = None): 
    obj.gmR, obj.tgmR, obj.etR, obj.tetR, = gm_type()
    obj.NR = np.linalg.inv(id2 - gmR @ tgmR)
    obj.tNR = np.linalg.inv(id2 - tgmR @ gmR)
    if phase != None:
        obj.gmR = obj.gmR*np.exp(1j*phase)
        obj.tgmR = obj.tgmR*np.exp(-1j*phase)
    
 
def set_up_gmL_from_file_alpha(obj,filename,phase = None):
    '''
    needs alpha to be defined globally 
    '''
    obj.gmL = np.zeros((2,2), dtype = complex)
    obj.tgmL = np.zeros((2,2), dtype = complex)
    obj.etL = np.zeros((2,2), dtype = complex)
    obj.tetL = np.zeros((2,2), dtype = complex)
    ym = np.load(filename)
    # gmL_old  = ym[0]
    # tgmL_old = ym[1]
    # etL_old  = ym[2]
    # tetL_old = ym[3]
    gmL_old  = ym[0] + 1j*ym[4]
    etL_old  = ym[1] + 1j*ym[5]
    tgmL_old = ym[2] + 1j*ym[6]
    tetL_old = ym[3] + 1j*ym[7]
    
    dx =  1/2*(- gmL_old[0,0] + gmL_old[1,1])
    dy = -1j/2*( gmL_old[0,0] + gmL_old[1,1])
    dx_new = dx*np.cos(alpha) + dy*np.sin(alpha)
    dy_new =-dx*np.sin(alpha) + dy*np.cos(alpha)
    obj.gmL[0,0] = -dx_new + 1j*dy_new
    obj.gmL[1,1] = dx_new +1j*dy_new
    obj.gmL[0,1] = gmL_old[0,1]
    obj.gmL[1,0] = gmL_old[1,0]
    #print('alpha',alpha) 
    # print('dx',dx)
    # print(dx_new)
    #  print('dy',dy)
    # print(dy_new)
    
    
    et_dx = 1/2*( - etL_old[0,0] + etL_old[1,1])
    et_dy = -1j/2*( etL_old[0,0] + etL_old[1,1])
    et_dx_new = et_dx*np.cos(alpha) + et_dy*np.sin(alpha)
    et_dy_new =-et_dx*np.sin(alpha) + et_dy*np.cos(alpha)
    obj.etL[0,0] = -et_dx_new + 1j*et_dy_new
    obj.etL[1,1] = et_dx_new +1j*et_dy_new
    obj.etL[0,1] = etL_old[0,1]
    obj.etL[1,0] = etL_old[1,0]
    
    tdx = 1/2*( - tgmL_old[0,0] + tgmL_old[1,1])
    tdy = +1j/2*( tgmL_old[0,0] + tgmL_old[1,1])
    tdx_new = tdx*np.cos(alpha) + tdy*np.sin(alpha)
    tdy_new = - tdx*np.sin(alpha) + tdy*np.cos(alpha)
    obj.tgmL[0,0] = -tdx_new - 1j*tdy_new
    obj.tgmL[1,1] = tdx_new - 1j*tdy_new
    obj.tgmL[0,1] = tgmL_old[0,1]
    obj.tgmL[1,0] = tgmL_old[1,0]
    
    tet_dx = 1/2*( - tetL_old[0,0] + tetL_old[1,1])
    tet_dy = +1j/2*( tetL_old[0,0] + tetL_old[1,1])
    tet_dx_new = tet_dx*np.cos(alpha) + tet_dy*np.sin(alpha)
    tet_dy_new =-tet_dx*np.sin(alpha) + tet_dy*np.cos(alpha)
    obj.tetL[0,0] = -tet_dx_new + 1j*tet_dy_new
    obj.tetL[1,1] = tet_dx_new +1j*tet_dy_new
    obj.tetL[0,1] = tetL_old[0,1]
    obj.tetL[1,0] = tetL_old[1,0]
    
    #print('et_dx_new',et_dx_new)
    #print('et_dy_new',et_dy_new)
    #print('etL:',obj.etL)
    #print('gmL:',obj.gmL)
    
    obj.NL = np.linalg.inv(id2 -   obj.gmL @ obj.tgmL)
    obj.tNL = np.linalg.inv(id2 - obj.tgmL @  obj.gmL)


# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:30:07 2023

@author: Sigrid Aunsmo

Functions to progress, save and load data
"""
import numpy as np
from numba import jit
#from functions import v_to_m, m_to_v

def v_to_m(v): 
    #global nm
    #nm = 4
    n1,n2 = np.shape(v)
    nm = int(n1/4)
    nx = n2
    return np.reshape(v,(nm,2,2,nx))

def m_to_v(m): 
    #global nm
    #nm = 4
    n1,n2,n3,n4 = np.shape(m)
    nm = n1
    nx = n4
    return np.reshape(m,(nm*2*2,nx))

#@jit(nopython = true)
def gamma_from_y(ym): 
    # gm  = ym[0]
    # et  = ym[1]
    # tgm = ym[2]
    # tet = ym[3]
    
    re_gm  = ym[0]
    re_et  = ym[1]
    re_tgm = ym[2]
    re_tet = ym[3]
    im_gm  = ym[4]
    im_et  = ym[5]
    im_tgm = ym[6]
    im_tet = ym[7]
    
    #print(im_et)
    #print(np.shape(im_et))
    gm  = re_gm  + 1j*im_gm
    tgm = re_tgm + 1j*im_tgm 
    et  = re_et  + 1j*im_et
    tet = re_tet + 1j*im_tet
    
    N = np.zeros_like(gm)
    tN = np.zeros_like(gm)
    
    for i in range(len(gm[0,0])): 
        N[:,:,i]  = np.linalg.inv(np.eye(2,dtype=complex) - gm[:,:,i] @ tgm[:,:,i] )
        tN[:,:,i] = np.linalg.inv(np.eye(2,dtype=complex) - tgm[:,:,i] @ gm[:,:,i])
    return gm,et,tgm,tet,N,tN


def calc_f_from_gamma(ym): 
    gm,et,tgm,tet, N, tN = gamma_from_y(ym)
    
    f = 2*np.einsum('ijk,jlk-> ilk', N , gm)
    
    fs = 1/2*( f[0,1] - f[1,0] ) 
    dz = 1/2*( f[0,1] + f[1,0] )
    dx = 1/2*( -f[0,0]+ f[1,1] )
    dy = -1j/2*( f[0,0] + f[1,1])
    
    tf = 2*np.einsum('ijk,jlk-> ilk', tN , tgm)
    
    tfs = 1/2*( tf[0,1] - tf[1,0] ) 
    tdz = 1/2*( tf[0,1] + tf[1,0] )
    tdx = 1/2*( -tf[0,0]+ tf[1,1] )
    tdy = 1j/2*( tf[0,0] + tf[1,1])

    return fs, dx, dy, dz, tfs, tdx, tdy, tdz 

def currents(ym): 

    gm,et,tgm,tet, N, tN = gamma_from_y(ym)
    
    f = 2*np.einsum('ijk,jlk-> ilk', N , gm)
    tf = 2*np.einsum('ijk,jlk-> ilk', tN , tgm)
    
    
    fs = 1/2*( f[0,1] - f[1,0] ) 
    dz = 1/2*( f[0,1] + f[1,0] )
    dx = 1/2*( -f[0,0]+ f[1,1] )
    dy = 1/(2*1j)*( f[0,0] + f[1,1])
    tfs = 1/2*( tf[0,1] - tf[1,0] ) 
    tdz = 1/2*( tf[0,1] + tf[1,0] )
    tdx = 1/2*( -tf[0,0]+ tf[1,1] )
    tdy = -1/(2*1j)*( tf[0,0] + tf[1,1])
    
    dN = np.zeros_like(gm, dtype = complex)
    dtN = np.zeros_like(gm, dtype = complex)
    for i in range(len(gm[0,0])): 
        #dN[:,:,i]  = np.eye(2,dtype=complex) - et[:,:,i] @ tet[:,:,i] 
        #dtN[:,:,i] = np.eye(2,dtype=complex) - tet[:,:,i] @ et[:,:,i]
        dN[:,:,i] = N[:,:,i] @ (et[:,:,i] @ tgm[:,:,i] + gm[:,:,i] @ tet[:,:,i] ) @ N[:,:,i]
        dtN[:,:,i] = tN[:,:,i] @ (tet[:,:,i] @ gm[:,:,i] + tgm[:,:,i] @ et[:,:,i] ) @ tN[:,:,i]
    #dN = np.einsum('ijk,jlk->ilk' ,N, np.einsum('ijk,jlk->ilk' , np.einsum('ijk,jlk->ilk',et,tgm) + np.einsum('ijk,jlk->ilk',gm,tet)    ,N )  )
    #dtN = np.einsum('ijk,jlk->ilk' ,tN, np.einsum('ijk,jlk->ilk' , np.einsum('ijk,jlk->ilk',tet,gm) + np.einsum('ijk,jlk->ilk',tgm,et)  , tN )  )
    
    
    #print(dN)
    #print(dN_at2)
    
    #return gm,et,tgm,tet, N, tN
    #df =  2*np.einsum('ijk,jlk-> ilk', dN   , et  )
    #dtf = 2*np.einsum('ijk,jlk-> ilk', dtN  , tet )
    '''Begge metodene fungerer visst'''
    df  = 2* np.einsum('ijk,jlk->ilk',N,np.einsum('ijk,jlk->ilk',et + np.einsum('ijk,jlk->ilk',gm,np.einsum('ijk,jlk->ilk',tet,gm)),tN)) 
    dtf = 2* np.einsum('ijk,jlk->ilk',tN,np.einsum('ijk,jlk->ilk',tet + np.einsum('ijk,jlk->ilk',tgm,np.einsum('ijk,jlk->ilk',et,tgm)),N)) 
    
    
    dfs = 1/2*( df[0,1] - df[1,0] ) 
    ddz = 1/2*( df[0,1] + df[1,0] )
    ddx = 1/2*( -df[0,0]+ df[1,1] )
    ddy = 1/(2*1j)*( df[0,0] + df[1,1])
    dtfs = 1/2*( dtf[0,1] - dtf[1,0] ) 
    dtdz = 1/2*( dtf[0,1] + dtf[1,0] )
    dtdx = 1/2*( -dtf[0,0]+ dtf[1,1] )
    dtdy = -1/(2*1j)*( dtf[0,0] + dtf[1,1])
    
    
    ISz_simp = np.imag( dx*dtdy - dy*dtdx + tdx*ddy - tdy*ddx )
    ISx_simp = np.imag( dy*dtdz - dz*dtdy + tdy*ddz - tdz*ddy )
    ISy_simp = np.imag( dz*dtdx - dx*dtdz + tdz*ddx - tdx*ddz )
    
    nn,nm,nx = np.shape(gm)
    g  = np.zeros_like(gm,dtype= complex)  # 2x2 g
    dg = np.zeros_like(gm,dtype= complex)  # 2x2 g
    
    for i in range(nx):
        g[:,:,i] = 2*N[:,:,i] - np.eye(2,dtype = complex)
        dg[:,:,i] = 2*dN[:,:,i] - np.eye(2,dtype = complex)
    
    
    id2 = np.eye(2,dtype = complex)
    id2nx = np.zeros((2,2,nx),dtype= complex)
    id2nx[0,0,:] = 1
    id2nx[1,1,:] = 1    
    
    rhotre = np.diag((1 + 0j,1 + 0j,-1 + 0j,-1 + 0j))
    
    gR = np.zeros((4,4,nx),dtype= complex)
    
    for i in range(nx):     
        #gR[0:2,0:2,i] = 2*N[:,:,i] - id2
        gR[0:2,2:4,i] = 2*N[:,:,i] @ gm[:,:,i]   #2*np.einsum('ijk,jlk->ilk',N[:],gm)
        #gR[2:4,2:4,i] = -2*tN[:,:,i] + id2
        gR[2:4,0:2,i] = -2*tN[:,:,i] @ tgm[:,:,i]  #-2*np.einsum('ijk,jlk->ilk',tN,tgm)
        gR[0:2,0:2,i] = id2
        gR[2:4,2:4,i] = - id2
        
    gA = np.zeros_like(gR, dtype = complex)
    for i in range(nx):
        gA[:,:,i]= - rhotre @ np.conj(gR[:,:,i].T) @ rhotre    
    
    dgR = np.zeros((4,4,nx),dtype = complex)
    '''Skal jeg her bruke '''
    for i in range(nx): 
        #dgR[0:2,0:2,i] = 2*dN[:,:,i] - id2   #2*dN 
        #dgR[0:2,2:4,i] = 2*dN[:,:,i] @ et[:,:,i] #2* ( np.einsum('ijk,jlk->ilk', N , np.einsum('ijk,jlk->ilk', et - np.einsum('ijk,jlk->ilk', gm, np.einsum('ijk,jlk->ilk',tet,gm) )  , tN ) ) )
        #dgR[2:4,2:4,i] = -(2*dtN[:,:,i] -id2)  #-2*dtN 
        #dgR[2:4,0:2,i] = -2*dtN[:,:,i] @ tet[:,:,i]   #-2*( np.einsum('ijk,jlk->ilk', tN , np.einsum('ijk,jlk->ilk', tet - np.einsum('ijk,jlk->ilk', tgm, np.einsum('ijk,jlk->ilk',et,tgm) )  , N ) ) )
        
        #dgR[0:2,0:2,i] = 2*dN[:,:,i]
        dgR[0:2,2:4,i] = df[:,:,i]
        dgR[2:4,0:2,i] = -dtf[:,:,i]
        #dgR[2:4,2:4,i] = -2*dtN[:,:,i]
    
    dgA = np.zeros_like(dgR, dtype = complex)
    
    for i in range(nx):
        dgA[:,:,i]= - rhotre @ np.conj(dgR[:,:,i].T) @ rhotre    
      

    sigmax = np.array([[0,1 +0j],[1,0 ]])
    sigmay = np.array([[0,-1j  ],[1j,0]])
    sigmaz = np.array([[1+0j,0],[0,-1]])
    sigmarhox = np.zeros((4,4),dtype = complex)
    sigmarhoy = np.zeros((4,4),dtype = complex)
    sigmarhoz = np.zeros((4,4),dtype = complex)
    sigmarhox[0:2,0:2] = sigmax
    sigmarhox[2:4,2:4] = -np.conj(sigmax)
    sigmarhoy[0:2,0:2] = sigmay
    sigmarhoy[2:4,2:4] = -np.conj(sigmay)
    sigmarhoz[0:2,0:2] = sigmaz
    sigmarhoz[2:4,2:4] = -np.conj(sigmaz)
    
    
    
    I = np.trace( np.einsum('ij,jkl->ikl' ,rhotre, np.einsum('ijk,jlk->ilk', gR, dgR) - np.einsum('ijk,jlk->ilk',gA, dgA) )   )
    ISx = np.trace( np.einsum('ij,jkl->ikl' ,sigmarhox, np.einsum('ijk,jlk->ilk', gR, dgR) - np.einsum('ijk,jlk->ilk',gA, dgA) )   )
    ISy = np.trace( np.einsum('ij,jkl->ikl' ,sigmarhoy, np.einsum('ijk,jlk->ilk', gR, dgR) - np.einsum('ijk,jlk->ilk',gA, dgA) )   )
    ISz = np.trace( np.einsum('ij,jkl->ikl' ,sigmarhoz, np.einsum('ijk,jlk->ilk', gR, dgR) - np.einsum('ijk,jlk->ilk',gA, dgA) )   )
    
    I_simp = np.real(2*(fs*dtfs - tfs*dfs - dz*dtdz + tdz*ddz - dx*dtdx + tdx*ddx - dy*dtdy + tdy*ddy))
    
    I_fs = np.real( fs*dtfs - tfs*dfs)
    I_dx = np.real(-dx*dtdx + tdx*ddx)
    I_dy = np.real(-dy*dtdy + tdy*ddy)
    I_dz = np.real(-dz*dtdz + tdz*ddz)
    
    return I, ISx, ISy, ISz, I_simp, ISx_simp, ISy_simp, ISz_simp, dfs,ddx,ddy,ddz,dtfs,dtdx,dtdy,dtdz, I_fs, I_dx, I_dy, I_dz
    
def current_2(ym): 
    gm  = ym[0] + 1j*ym[4]
    et  = ym[1] + 1j*ym[5]
    tgm = ym[2] + 1j*ym[6]
    tet = ym[3] + 1j*ym[7]
    id2 = np.eye(2,dtype = complex)
    
    
    N  = np.zeros(np.shape(gm),dtype = complex) 
    tN = np.zeros(np.shape(gm),dtype = complex)
    for i in range(len(N[0,0])):
        N[:,:,i]  = np.linalg.inv(id2 - gm[:,:,i]  @ tgm[:,:,i] )
        tN[:,:,i] = np.linalg.inv(id2 - tgm[:,:,i] @  gm[:,:,i] )
        
    f = 2*np.einsum('ijk,jlk->ilk',   N,  gm)
    tf = 2*np.einsum('ijk,jlk->ilk', tN, tgm)
    
    et_p_gm_tet_gm = et + np.einsum('ijk,jlk->ilk',gm, np.einsum('ijk,jlk->ilk',tet,gm))
    tet_p_tgm_et_tgm = tet +  np.einsum('ijk,jlk->ilk',tgm, np.einsum('ijk,jlk->ilk',et,tgm))
    df  = 2*np.einsum('ijk,jlk->ilk',  N, np.einsum('ijk,jlk->ilk',   et_p_gm_tet_gm ,tN))
    dtf = 2*np.einsum('ijk,jlk->ilk', tN, np.einsum('ijk,jlk->ilk', tet_p_tgm_et_tgm , N))
    
    fs = 1/2*(f[0,1] - f[1,0])
    dfs = 1/2*(df[0,1] - df[1,0]) 
    tfs = 1/2*(tf[0,1] - tf[1,0])
    dtfs = 1/2*(dtf[0,1] - dtf[1,0])
    
    ft = 1/2*(  f[0,1] +  f[1,0] )
    dft = 1/2*(df[0,1] + df[1,0] )
    tft = 1/2*(tf[0,1] + tf[1,0])
    dtft= 1/2*(dtf[0,1]+dtf[1,0])
    
    dx = 1/2*( - f[0,0] + f[1,1])
    ddx = 1/2*( - df[0,0] + df[1,1])
    tdx = 1/2*( - tf[0,0] + tf[1,1])
    dtdx = 1/2*( - dtf[0,0] + dtf[1,1])
    
    dy = -1j/2*( f[0,0] + f[1,1])
    ddy = -1j/2*( df[0,0] + df[1,1])
    tdy = 1j/2*( tf[0,0] + tf[1,1])
    dtdy = 1j/2*( dtf[0,0] + dtf[1,1])
    
    #I = np.real(2*fs*dtfs - 2*ft*dtft - f[0,0]*dtf[0,0] - f[1,1]*dtf[1,1] - 2*tfs*dfs + 2*tft*dft + tf[0,0]*df[0,0] + tf[1,1]*df[1,1])
    I = 2*np.real(fs*dtfs - tfs*dfs - ft*dtft + tft*dft - dx*dtdx + tdx*ddx - dy*dtdy + tdy*ddy)
    
    
    return I 
    
    
def save_single_E(sol,E,xnx,filename):
    'input: y on matrix form'
    y  = sol.sol(xnx) 
    ym = v_to_m(y)
    gm,et,tgm,tet, N, tN = gamma_from_y(ym)
    fs, dx, dy, dz, tfs, tdx, tdy, tdz = calc_f_from_gamma(ym)
    #x = sol.x
    x = xnx
    ynx = sol.sol(xnx)
    J, Jsx, Jsy, Jsz, I_simp, ISx_simp, ISy_simp, ISz_simp,dfs,ddx,ddy,ddz,dtfs,dtdx,dtdy,dtdz,I_fs, I_dx, I_dy, I_dz = currents(ym)
    DOS = calc_dos(gm,tgm)
    M = magnetization_one_E(ym)
    Mx = M[0]
    My = M[1]
    Mz = M[2]
    xnx = sol.x
    I2 = current_2(ym)
    np.savez('data/'+filename,ym,fs,dx,dy,dz,x,gm,N,E,ynx,tfs, tdx, tdy, tdz, tgm, et, tet, tN, J, Jsx, Jsy, Jsz, DOS, Mx, My, Mz, I_simp, ISx_simp, ISy_simp, ISz_simp,dfs,ddx,ddy,ddz,dtfs,dtdx,dtdy,dtdz,xnx,I2, I_fs, I_dx, I_dy, I_dz)



def load_single_E(filename):
    'returning the data as a class'
    loaded_data = np.load('data/'+filename)
    
    data = Data_single_E(loaded_data)
    return data


class Data_single_E:
    def __init__(self,ld): #input: loaded data
        self.y = ld['arr_0']
        self.fs = ld['arr_1']
        self.dx = ld['arr_2']
        self.dy = ld['arr_3']
        self.dz = ld['arr_4']
        self.x = ld['arr_5']
        self.gm = ld['arr_6']
        self.N = ld['arr_7']
        self.E = ld['arr_8']
        self.ynx = ld['arr_9']
        self.tfs = ld['arr_10']
        self.tdx = ld['arr_11']
        self.tdy = ld['arr_12']
        self.tdz = ld['arr_13']
        self.tgm    =ld['arr_14']
        self.et     =ld['arr_15']   
        self.tet    =ld['arr_16']
        self.tN     =ld['arr_17']
        self.J    = ld['arr_18']
        self.Jsx  = ld['arr_19']
        self.Jsy  = ld['arr_20']
        self.Jsz  = ld['arr_21']
        self.DOS  = ld['arr_22']
        self.Mx   = ld['arr_23']
        self.My   = ld['arr_24']
        self.Mz   = ld['arr_25']
        self.J_simp = ld['arr_26']
        self.Jsx_simp = ld['arr_27']
        self.Jsy_simp = ld['arr_28']
        self.Jsz_simp = ld['arr_29']
        self.dfs  = ld['arr_30']
        self.ddx  = ld['arr_31']
        self.ddy  = ld['arr_32']
        self.ddz  = ld['arr_33']
        self.dtfs = ld['arr_34']
        self.dtdx = ld['arr_35']
        self.dtdy = ld['arr_36']
        self.dtdz = ld['arr_37']
        self.xnx  = ld['arr_38']
        self.I2 = ld['arr_39']
        self.I_fs = ld['arr_40'] 
        self.I_dx = ld['arr_41']
        self.I_dy = ld['arr_42']
        self.I_dz = ld['arr_43']
        
def save_multipleE(E_list,DOS0,DOSL,filename):
    
    
    np.savez(filename,E_list,DOS0,DOSL)
    

def load_multiple_E(filename):
    loaded_data = np.load(filename)
    data = Data_multiple_E(loaded_data)
    return data
    
class Data_multiple_E: 
    def __init__(self,ld): 
        self.E_list = ld['arr_0']
        self.DOS0 = ld['arr_1']
        self.DOSL = ld['arr_2']

def save_multiple_E_new(filename,Es,x): 
    dEs = np.zeros_like(Es)
    dEs[:-1] = Es[1:] - Es[:-1] 
    dEs[-1] = dEs[-2]
    Mx  = np.zeros(np.shape(x), dtype= complex) 
    My  = np.zeros_like(Mx)
    Mz  = np.zeros_like(Mx)
    J   = np.zeros_like(Mx)
    Jsz = np.zeros_like(Mx)
    Jsx = np.zeros_like(Mx)
    Jsy = np.zeros_like(Mx)
    DOS = np.zeros_like(Mx)
    
    for i in range(len(Es)): 
        E = Es[i]
        dE = dEs[i]
        datai = load_single_E(filename+str(i)+'.npz')
        
        Mx += datai.Mx  *dE * np.tanh(np.real(E))
        My += datai.My  *dE * np.tanh(np.real(E))
        Mz += datai.Mz  *dE * np.tanh(np.real(E))
        J  += datai.J   *dE * np.tanh(np.real(E))
        Jsz+= datai.Jsz_simp *dE * np.tanh(np.real(E))
        Jsx+= datai.Jsx_simp *dE * np.tanh(np.real(E))
        Jsy+= datai.Jsy_simp *dE * np.tanh(np.real(E))
        DOS+= datai.DOS *dE * np.tanh(np.real(E))
    np.savez('data/'+filename+'.npz',Mx,My,Mz,J,Jsx,Jsy,Jsz,DOS)
    
    
class Data_multiple_E_new:
    def __init__(self,ld): 
        self.Mx  = ld['arr_0']
        self.My  = ld['arr_1']
        self.Mz  = ld['arr_2']
        self.J   = ld['arr_3']
        self.Jsx = ld['arr_4']
        self.Jsy = ld['arr_5']
        self.Jsz = ld['arr_6']
        self.DOS = ld['arr_7']

def load_multiple_E_new(filename):
    ld = np.load('data/'+filename+'.npz')
    data = Data_multiple_E_new(ld)
    return data
        
    
def calc_dos(gm,tgm): 
    g = np.zeros_like(gm)
    tg = np.zeros_like(gm)
    for i in range(len(g[0,0])):    
        g[:,:,i] = np.linalg.inv( np.eye(2,dtype = complex) -  gm[:,:,i] @ tgm[:,:,i] )
        tg[:,:,i] = np.linalg.inv(np.eye(2,dtype = complex) - tgm[:,:,i] @  gm[:,:,i] ) 
    return 1/4 * (np.trace(g) - np.trace(tg))
        
'''this one should maybe include the tanh(beta E/2) '''
def magnetization_one_E(ym): 
    fs, dx, dy, dz, tfs, tdx, tdy, tdz  = calc_f_from_gamma(ym)
    
    
    Mx = np.real(dx * tfs - tdx * fs)
    My = np.real(dy * tfs - tdy * fs)
    Mz = np.real(dz * tfs - tdz * fs)
    
    # gm,et,tgm,tet, N, tN = gamma_from_y(ym)
    # nn,nm,nx = np.shape(gm)
    
    
    # id2 = np.eye(2,dtype = complex)
    # id2nx = np.zeros((2,2,nx),dtype= complex)
    # id2nx[0,0,:] = 1
    # id2nx[1,1,:] = 1    
    
    # rhotre = np.diag((1 + 0j,1 + 0j,-1 + 0j,-1 + 0j))
    # #print(rhotre)
    
    
    # gR = np.zeros((4,4,nx),dtype= complex)
    
    # for i in range(nx):     
    #     gR[0:2,0:2,i] = 2*N[:,:,i] - id2
    #     gR[0:2,2:4,i] = 2*N[:,:,i] @ gm[:,:,i]   #2*np.einsum('ijk,jlk->ilk',N[:],gm)
    #     gR[2:4,2:4,i] = -(2*tN[:,:,i] - id2)
    #     gR[2:4,0:2,i] = -2*tN[:,:,i] @ tgm[:,:,i]  #-2*np.einsum('ijk,jlk->ilk',tN,tgm)
    
    # gA = np.zeros_like(gR, dtype = complex)
    # for i in range(nx):
    #     gA[:,:,i]= - rhotre @ np.conj(gR[:,:,i].T) @ rhotre    
    
    # sigmax = np.array([[0,1 +0j],[1,0 ]])
    # sigmay = np.array([[0,-1j  ],[1j,0]])
    # sigmaz = np.array([[1+0j,0],[0,-1]])
    
    # sigmarhox = np.zeros((4,4),dtype = complex)
    # sigmarhoy = np.zeros((4,4),dtype = complex)
    # sigmarhoz = np.zeros((4,4),dtype = complex)
    
    # sigmarhox[0:2,0:2] = sigmax
    # sigmarhox[2:4,2:4] = -np.conj(sigmax)
    
    # sigmarhoy[0:2,0:2] = sigmay
    # sigmarhoy[2:4,2:4] = -np.conj(sigmay)
    
    # sigmarhoz[0:2,0:2] = sigmaz
    # sigmarhoz[2:4,2:4] = -np.conj(sigmaz)
    
    
    # Mx = np.trace( np.einsum('jk,klm->jlm',sigmarhox, gR - gA)  )
    # My = np.trace( np.einsum('jk,klm->jlm',sigmarhoy, gR - gA)  )
    # Mz = np.trace( np.einsum('jk,klm->jlm',sigmarhoz, gR - gA)  )
    return np.array([Mx,My,Mz])
    
def save_mxmy(filename,sol,x): 
    y  = sol.sol(x) 
    ym = v_to_m(y)
    M = magnetization_one_E(ym)
    Mx = M[0]
    My = M[1]
    
    
    np.savez(f'data2/{filename}_mxmy.npz', Mx, My) 
    
    
def load_mxmy(filename): 
    ld = np.load(f'data2/{filename}_mxmy.npz')
    Mx  = ld['arr_0']
    My  = ld['arr_1']
    return Mx, My
    
    
def save_mxmy_fsdxdy(filename,sol,x): 
    y  = sol.sol(x) 
    ym = v_to_m(y)
    fs, dx, dy, dz, tfs, tdx, tdy, tdz  = calc_f_from_gamma(ym)
    M = magnetization_one_E(ym)
    Mx = M[0]
    My = M[1]
    #print('hello')
    np.savez(f'data2/{filename}_mxmy.npz', Mx, My, np.abs(fs), np.abs(dx),np.abs(dy)) 
    
    
def load_mxmy_fsdxdy(filename): 
    ld = np.load(f'data2/{filename}_mxmy.npz')
    Mx  = ld['arr_0']
    My  = ld['arr_1']
    fs  = ld['arr_2']
    dx  = ld['arr_3']
    dy  = ld['arr_4']
    return Mx, My, fs,dx,dy
    
    
    
    
def magnetization(ym): 
    sigmax = np.array([[0,1 +0j],[1,0 ]])
    sigmay = np.array([[0,-1j],[1j,0]])
    sigmaz = np.array([[1+0j,0],[0,-1]])
    
    gm,et,tgm,tet, N, tN = gamma_from_y(ym)
    g = 2*np.einsum('ijk,jlk->ilk',N,gm)
    tg = 2*np.einsum('ijk,jlk->ilk',tN,tgm)
     
    gdag = np.zeros_like(g)
    gdag[0,0] = np.conj(g[0,0]) 
    gdag[1,1] = np.conj(g[1,1])
    gdag[1,0] = np.conj(g[0,1])
    gdag[0,1] = np.conj(g[1,0])
    
    tgdag = np.zeros_like(g)
    tgdag[0,0] = np.conj(tg[0,0]) 
    tgdag[1,1] = np.conj(tg[1,1])
    tgdag[1,0] = np.conj(tg[0,1])
    tgdag[0,1] = np.conj(tg[1,0])
    
    mx_expr = np.einsum('ij,jkl->ikl',sigmax,g + gdag) + np.einsum('ij,jkl->ikl',sigmax,tg + tgdag)
    #mx =  np.trace(np.einsum('ij,jkl->ikl',sigmax,g + gdag) + np.einsum('ij,jkl->ikl',sigmax,tg + tgdag) )
    mx = mx_expr[0,0,:] + mx_expr[1,1,0]
    
    my =  np.trace(np.einsum('ij,jkl->ikl',sigmay,g + gdag) + np.einsum('ij,jkl->ikl',-sigmay,tg + tgdag) )
    mz =  np.trace(np.einsum('ij,jkl->ikl',sigmaz,g + gdag) + np.einsum('ij,jkl->ikl',sigmaz,tg + tgdag) )
    
    return mx,my,mz
    
    
    
    
    
    
    
    
        
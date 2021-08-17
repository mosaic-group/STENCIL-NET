import numpy as np
import torch
import os
from tqdm import tqdm_notebook as tqdm

def dflux(u):
    return u

def flux(u):
    return 0.5*(u**2)

def Kdv_flux(u):
    return 3.0*(u**2)

def dKdv_flux(u):
    return 6.0*u

def WENO_scheme(w,dx,sol):
    """Computation of WENO scheme."""
    if(sol=='burgers' or sol=='KS'):
        a=max(abs(dflux(w))); v=0.5*(flux(w)+a*w); u=np.roll(0.5*(flux(w)-a*w),[0,-1]);  #flux splitting
    if(sol=='kdv'):
        a=max(abs(dKdv_flux(w))); v=0.5*(Kdv_flux(w)+a*w); u=np.roll(0.5*(Kdv_flux(w)-a*w),[0,-1]);
    
    vmm = np.roll(v,[0,2]);  # w[i-2]
    vm  = np.roll(v,[0,1]);  # w[i-1]
    vp  = np.roll(v,[0,-1]); # w[i+1]
    vpp = np.roll(v,[0,-2]); # w[i+2]
    
    # Polynomials
    # S0 - {vmm, vm, v}, S1 = {vm,v,vp}, S2 = {v,vp,vpp}
    p0n = (2.*vmm - 7.*vm + 11.*v)/6.;
    p1n = ( -vm  + 5.*v  + 2.*vp)/6.;
    p2n = (2.*v   + 5.*vp - vpp )/6.;
    
    # Smoothness indicators
    B0n = (13./12)*((vmm-2.*vm+v  )**2) + (1./4)*((vmm-4.*vm+3.*v)**2); 
    B1n = (13./12)*((vm -2.*v +vp )**2) + (1./4)*((vm-vp)**2);
    B2n = (13./12)*((v  -2.*vp+vpp)**2) + (1./4)*((3.*v-4.*vp+vpp)**2);
    
    d0n = 1./10; d1n = 6./10; d2n = 3./10; 
    epsilon = 1e-6
    
    alpha0n = d0n/((epsilon + B0n)**2);
    alpha1n = d1n/((epsilon + B1n)**2);
    alpha2n = d2n/((epsilon + B2n)**2);
    alphasumn = alpha0n + alpha1n + alpha2n;
    
    w0n = alpha0n/alphasumn;
    w1n = alpha1n/alphasumn;
    w2n = alpha2n/alphasumn;

    Rplus = w0n*p0n + w1n*p1n + w2n*p2n
    
    # here u[i] is w[i+1]
    umm = np.roll(u,[0,2]);  # w[i-1]
    um  = np.roll(u,[0,1]);  # w[i]
    up  = np.roll(u,[0,-1]); # w[i+2]
    upp = np.roll(u,[0,-2]); # w[i+3]
    
    # S0 = {upp, up, u}, S1 = {up, u, um}, S2 = {u, um, umm}
    p0p = ( -umm + 5*um + 2*u  )/6.; # S2
    p1p = ( 2*um + 5*u  - up   )/6.; # S1
    p2p = (11*u  - 7*up + 2*upp)/6.; # S0
    
    B0p = (13./12)*((umm-2.*um+u  )**2) + (1./4)*((umm-4.*um+3.*u)**2); 
    B1p = (13./12)*((um -2.*u +up )**2) + (1./4)*((um-up)**2);
    B2p = (13./12)*((u  -2.*up+upp)**2) + (1./4)*((3.*u -4.*up+upp)**2);

    d0p = 3./10; d1p = 6./10; d2p = 1./10; epsilon = 1e-6;
    alpha0p = d0p/((epsilon + B0p)**2);
    alpha1p = d1p/((epsilon + B1p)**2);
    alpha2p = d2p/((epsilon + B2p)**2);
    alphasump = alpha0p + alpha1p + alpha2p;

    w0p = alpha0p/alphasump;
    w1p = alpha1p/alphasump;
    w2p = alpha2p/alphasump;
    
    Rminus = w0p*p0p + w1p*p1p + w2p*p2p;
    
    # ((f+_{i+0.5} + f-_{i+0.5}) - (f+_{i-0.5} + f-_{i-0.5}))/(dx) -- flux approximation
    return (Rplus + Rminus - np.roll(Rplus,[0,1]) - np.roll(Rminus,[0,1]))/(dx)

def burgers_simulation(Tsim, Lx, x, D, dt, A, w, phi, l, N, L):
    """
    Computes simulation data for Forced-Burgers equation.
    
    Keyword arguments:
    Tsim -- number of timesteps to be simulated
    Lx -- number spatial grid points
    x -- spatial grid points
    D -- diffusion constant
    dt -- length of timesteps
    A, w, phi, l, N, L -- parameters forcing terms
    """
    dx = x[1] - x[0]
    N  = len(A)
                                                                                      
    u_EN5_rk3  = np.zeros((Lx,Tsim))
    phase_uEN5 = np.zeros((Lx,Tsim))
    
    u_EN5_rk3[:,0] = np.exp(-(x-3)**2)
                                                                                      
    time = 0
    zf   = 1.0
    for j in tqdm(range(0,Tsim-1)):
        # computing forcing for burgers
        forcing = np.zeros((Lx,))
        for k in range(0, N):
            forcing = zf*forcing + zf*A[k]*np.sin(w[k]*time + 2.0*np.pi*l[k]*(x/L) + phi[k])

        um = np.roll(u_EN5_rk3[:,j],[0,1])
        up = np.roll(u_EN5_rk3[:,j],[0,-1])
        diff = D*(up - 2.0*u_EN5_rk3[:,j] + um)/(dx*dx)
        phase_uEN5[:,j] = diff - WENO_scheme(u_EN5_rk3[:,j],dx,sol='burgers')
        k1 = dt*(phase_uEN5[:,j]) + dt*forcing
        temp = u_EN5_rk3[:,j] + 0.5*k1

        forcing = np.zeros((Lx,))
        for k in range(0, N):
            forcing = zf*forcing + zf*A[k]*np.sin(w[k]*(time + 0.5*dt) + 2.0*np.pi*l[k]*(x/L) + phi[k])

        um = np.roll(temp,[0,1])
        up = np.roll(temp,[0,-1])
        diff = D*(up - 2.0*temp + um)/(dx*dx)

        k2 = dt*diff - dt*WENO_scheme(temp,dx,sol='burgers') + dt*forcing
        temp = u_EN5_rk3[:,j] - k1 + 2.0*k2

        forcing = np.zeros((Lx,))
        for k in range(0, N):
            forcing = zf*forcing + zf*A[k]*np.sin(w[k]*(time + dt) + 2.0*np.pi*l[k]*(x/L) + phi[k])

        um = np.roll(temp,[0,1])
        up = np.roll(temp,[0,-1])
        diff = D*(up - 2.0*temp + um)/(dx*dx)

        k3 = dt*diff - dt*WENO_scheme(temp,dx,sol='burgers') + dt*forcing
        u_EN5_rk3[:,j+1] = u_EN5_rk3[:,j] + (1./6.)*(k1 + 4.0*k2 + k3)

        time = time + dt
    
    return u_EN5_rk3, phase_uEN5

def forcing_terms(A, w, phi, l, L, Lxc, T, Ltc, N, dtc):
    """Computes coarse forcing terms."""
    x = np.linspace(0,L,Lxc); t = np.linspace(0,T,Ltc)
    XX, TT = np.meshgrid(x,t); xx = XX.T; tt = TT.T
    zf = 1.0
    Fc = np.zeros((Lxc,Ltc))
    for k in range(0, N):
        Fc = zf*Fc + zf*A[k]*np.sin(w[k]*tt + 2.0*np.pi*l[k]*(xx/L) + phi[k])   
    
    shift = 0.5*dtc
    x = np.linspace(0,L,Lxc); t = np.linspace(0 + shift, T + shift,Ltc)
    XX, TT = np.meshgrid(x,t); xx = XX.T; tt = TT.T
    Fc_0p5 = np.zeros((Lxc,Ltc))
    for k in range(0, N):
        Fc_0p5 = zf*Fc_0p5 + zf*A[k]*np.sin(w[k]*tt + 2.0*np.pi*l[k]*(xx/L) + phi[k])

    shift = dtc
    x = np.linspace(0,L,Lxc); t = np.linspace(0 + shift, T + shift,Ltc)
    XX, TT = np.meshgrid(x,t); xx = XX.T; tt = TT.T
    Fc_p1 = np.zeros((Lxc,Ltc))
    for k in range(0, N):
        Fc_p1 = zf*Fc_p1 + zf*A[k]*np.sin(w[k]*tt + 2.0*np.pi*l[k]*(xx/L) + phi[k])

    shift = 0.5*dtc
    x = np.linspace(0,L,Lxc); t = np.linspace(0 - shift, T - shift,Ltc)
    XX, TT = np.meshgrid(x,t); xx = XX.T; tt = TT.T
    Fc_0m5 = np.zeros((Lxc,Ltc))
    for k in range(0, N):
        Fc_0m5 = zf*Fc_0m5 + zf*A[k]*np.sin(w[k]*tt + 2.0*np.pi*l[k]*(xx/L) + phi[k])
    
    shift = dtc
    x = np.linspace(0,L,Lxc); t = np.linspace(0 - shift, T - shift,Ltc)
    XX, TT = np.meshgrid(x,t); xx = XX.T; tt = TT.T
    Fc_m1 = np.zeros((Lxc,Ltc))
    for k in range(0, N):
        Fc_m1 = zf*Fc_m1 + zf*A[k]*np.sin(w[k]*tt + 2.0*np.pi*l[k]*(xx/L) + phi[k])
        
    return Fc, Fc_0p5, Fc_p1, Fc_0m5, Fc_m1

def noise_initialization(u_coarse_noise, lam):
    """
    Computes an initial noise estimate of the measurement data using Tikhoniv regularization.
    
    Keyword arguments:
    u_coarse_noise -- the noisy measurement data
    lam -- tunable parameter regularizing smoothness
    """
    Lxc, Ltc = u_coarse_noise.shape
    
    m = Lxc
    DD = np.zeros((m,m))
    DD[0,:4] = [2,-5,4,-1]
    DD[m-1,m-4:] = [-1,4,-5,2]
    for i in range(1,m-1):
        DD[i,i] = -2
        DD[i,i+1] = 1
        DD[i,i-1] = 1
    DD = DD.dot(DD)

    u_smooth = np.zeros((Lxc,Ltc))
    for time in range(Ltc):
        u_smooth[:,time] = np.linalg.solve(np.eye(m) + lam*DD.T.dot(DD), u_coarse_noise[:,time])
        
    return u_coarse_noise - u_smooth

def load_denoising_model(s, t, nn, device):
    """
    Loads a pre-trained model for KdV signal-noise decomposition.
    
    Keyword arguments:
    s -- spatial subsampling factor
    t -- temporal subsampling factor
    nn -- number of neurons
    device -- torch.device to map output into
    """
    for path in os.listdir("models/"):
        if "kdv" in path and "Lxc{}".format(s) in path and "Ltc{}".format(t) in path and "_{}_".format(nn) in path:
            net = torch.load("models/" + path, map_location=device)
            return net
        
    raise FileNotFoundError("No model for given configuration! Please check description of possible pretrained configurations.")

def load_simulation_model(s, t, nn, device):
    """
    Loads a pre-trained model for forced Burgers simulation.
    
    Keyword arguments:
    s -- spatial subsampling factor
    t -- temporal subsampling factor
    nn -- number of neurons
    device -- torch.device to map output into
    """
    for path in os.listdir("models/"):
        if "burgers" in path and "Lxc{}".format(s) in path and "Ltc{}".format(t) in path and "_{}_".format(nn) in path:
            net = torch.load("models/" + path, map_location=device)
            return net
        
    raise FileNotFoundError("No model for given configuration! Please check description of possible pretrained configurations.")
    
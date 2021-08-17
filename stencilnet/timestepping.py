import torch

def forward_rk3_error(net, target, dt, m, wd, fc=None, fc_0p5=None, fc_p1=None):
    """
    Computes MSE for predicting solution forward in time using RK3 with possible forcing terms.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    noise -- estimated noise of measurement (default = None)
    fc -- forcing terms at current timestep (default = None)
    fc_0p5 -- forcing terms half a timestep into the future (default = None)
    fc_1p -- forcing terms one timestep into the future (default = None)
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize forcing terms
    fc     = fc if fc is not None else torch.zeros_like(target)
    fc_0p5 = fc_0p5 if fc_0p5 is not None else torch.zeros_like(target)
    fc_p1  = fc_p1 if fc_p1 is not None else torch.zeros_like(target)
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[0:-m,:])
    p_old  = pred[0:-m,:].clone()
    
    for j in range(m-1):
        k1    = dt*(net(p_old) + fc[j:-m+j,:])        # dt*f(t,y^n)
        temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
        k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     # dt*f(t+0.5*dt, y^n + 0.5*k1)
        temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2.0*k2
        k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      # dt*f(t+dt, y^n - k1 + 2.0*k2)
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4.0*k2 + k3)
        res   = res + wd[j+1]*((target[j+1:-m+j+1,:] - (p_new + noise[j+1:-m+j+1,:]))**2)
        p_old = p_new
        
    k1    = dt*(net(p_old) + fc[j:-m+j,:])        
    temp  = p_old + 0.5*k1                           
    k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     
    temp  = p_old - k1 + 2.0*k2                      
    k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      
    p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
    res   = res +  wd[m]*((target[m:,:] - (p_new + noise[m:,:]))**2)
    
    return torch.mean(res)

def backward_rk3_error(net, target, dt, m, wd, fc=None, fc_0m5=None, fc_m1=None):
    """
    Computes MSE for predicting solution backward in time using RK3 with possible forcing terms.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    noise -- estimated noise of measurement (default = None)
    fc -- forcing terms at current timestep (default = None)
    fc_0m5 -- forcing terms half a timestep into the past (default = None)
    fc_1m -- forcing terms one timestep into the past (default = None)
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize forcing terms
    fc     = fc if fc is not None else torch.zeros_like(target)
    fc_0m5 = fc_0m5 if fc_0m5 is not None else torch.zeros_like(target)
    fc_m1  = fc_m1 if fc_m1 is not None else torch.zeros_like(target)
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[m:,:])
    p_old  = pred[m:,:].clone()
    
    k1    = -dt*(net(p_old) + fc[m:,:])
    temp  = p_old + 0.5*k1
    k2    = -dt*(net(temp) + fc_0m5[m:,:])
    temp  = p_old - k1 + 2.0*k2
    k3    = -dt*(net(temp) + fc_m1[m:,:])
    p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
    res   = res + wd[1]*((target[m-1:-1,:] - (p_new + noise[m-1:-1,:]))**2)
    p_old = p_new
    
    for j in range(1,m):
        k1    = -dt*(net(p_old) + fc[m-j:-j,:])       # -dt*f(t,y^n)
        temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
        k2    = -dt*(net(temp) + fc_0m5[m-j:-j,:])    # -dt*f(t+0.5*dt, y^n + 0.5*k1)
        temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2*k2
        k3    = -dt*(net(temp) + fc_m1[m-j:-j,:])     # -dt*f(t+dt, y^n - k1 + 2.0*k2)
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4*k2 + k3)
        res   = res + wd[j+1]*((target[m-(j+1):-(j+1),:] - (p_new + noise[m-(j+1):-(j+1),:]))**2)
        p_old = p_new
    
    return torch.mean(res)

def forward_rk3_tvd_error(net, target, dt, m, wd):
    """
    Computes MSE for predicting solution forward in time using RK3 TVD.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[0:-m,:])
    p_old  = pred[0:-m,:].clone()
    
    for j in range(m-1):
        u1    = p_old + dt*net(p_old)
        u2    = 0.75*p_old.clone() + 0.25*u1.clone() + 0.25*dt*net(u1)
        p_new = (1./3.)*p_old.clone() + (2./3.)*u2.clone() + (2./3.)*dt*net(u2)
        res   = res + wd[j+1]*((target[j+1:-m+j+1,:] - (p_new + noise[j+1:-m+j+1,:]))**2)
        p_old = p_new
        
    u1    = p_old + dt*net(p_old)
    u2    = 0.75*p_old.clone() + 0.25*u1.clone() + 0.25*dt*net(u1)
    p_new = (1./3.)*p_old.clone() + (2./3.)*u2.clone() + (2./3.)*dt*net(u2)
    res   = res + wd[m]*((target[m:,:] - (p_new + noise[m:,:]))**2)
    
    return torch.mean(res)

def backward_rk3_tvd_error(net, target, dt, m, wd):
    """
    Computes MSE for predicting solution backward in time using RK3 TVD.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[m:,:])
    p_old  = pred[m:,:].clone()
    
    for j in range(m):
        u1    = p_old - dt*net(p_old)
        u2    = 0.75*p_old.clone() + 0.25*u1.clone() - 0.25*dt*net(u1)
        p_new = (1./3.)*p_old.clone() + (2./3.)*u2.clone() - (2./3.)*dt*net(u2)
        res   = res + wd[j+1]*((target[m-(j+1):-(j+1),:] - (p_new + noise[m-(j+1):-(j+1),:]))**2)
        p_old = p_new
    
    return torch.mean(res)                        
                             
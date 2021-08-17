import torch
import torch.nn as nn

class MLPConv(nn.Module):
    """
    MLPConv unit for STENCIL-NET.
    
    Keyword arguments:
    sizes -- layer sizes
    noise -- initial noise estimate for noisy data (default=None)
    seed -- seed for random network initialization (default=0)
    fs -- size of filters (default=7)
    activation -- activation function to be applied after linear transformations (default=torch.nn.ELU())
    """
    
    def __init__(self, sizes, noise=None, seed=0, fs=7, activation=nn.ELU()):
        super(MLPConv, self).__init__()
        
        torch.manual_seed(seed)
        
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        
        self.fs    = fs
        self.sig   = activation
        self.layer = nn.ModuleList()
        
        for i in range(len(sizes)-1):
            linear = nn.Linear(in_features=sizes[i], out_features=sizes[i+1])
            
            print("input", sizes[i], "output", sizes[i+1])
            nn.init.xavier_normal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)
            
            self.layer.append(linear)
            
        self.noise = None if noise is None else nn.Parameter(noise, requires_grad=True)

    def forward(self, x):
        x = self._preprocess(x)
        for i, layer in enumerate(self.layer):
            x = layer(x)
            if i < len(self.layer) - 1:
                x = self.sig(x)
        
        return x.squeeze()
    
    def _preprocess(self, x):
        """Prepares filters for forward pass."""
        x  = x.unsqueeze(-1)
        px = x.clone()
        
        for i in range(1, int(self.fs/2)+1):
            r = torch.roll(x, (-1)*i, 1)
            l = torch.roll(x, i, 1)
            
            px = torch.cat([l, px, r], -1)
        
        return px
        
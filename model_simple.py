import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from util.pscan import selective_scan_fn

from einops.layers.torch import Reduce

from collections import namedtuple
from util.patch_embed import PatchEmbed
import torch.nn.functional as F
from util.pscan import selective_scan_fn



class SSM(nn.Module):
    def __init__(self,
                 dim = 256,
                 dt_rank = 32,
                 dim_inner = 512,
                 d_state = 256,
                 ):
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # proj to B_t, C_t, d_t
        self.proj_bc = nn.Linear(dim, dt_rank + 2*d_state,bias=False)
        # proj dt to A
        self.dt_proj = nn.Linear(d_state,dim_inner,bias = True) 

        # defind A_log and D as parameters
        # A: dim_inner x d_state

        A = torch.arange(1,d_state+1,dtype=torch.float32).repeat(dim_inner,1)
        A_log = torch.log(A)
        self.A_log = nn.parameter(A_log) # reparametrizition trick: https://arxiv.org/abs/2311.14495
        self.D = nn.parameter(torch.ones(dim_inner)) 

        self.selective_scan_cuda = selective_scan_fn

    def forward(self,x,z):
        # x: B x L x dim
        A = -torch.exp(self.A_log.float()) # dim_inner x d_state
        D = self.D.float()  # dim_inner (bias)

        deltaBC = self.proj_bc(x) 
        dt, B, C = torch.split(deltaBC,[self.dt_rank,self.d_state,self.d_state],dim=-1)
        dt = self.dt_proj.weight @ dt # B x L x dim_inner # do not add bias here

        dt = rearrange(dt,'b l d -> b d l')
        x = rearrange(x,'b l d -> b d l')
        B = rearrange(B,'b l d -> b d l')
        C = rearrange(C,'b l d -> b d l')
        z = rearrange(z,'b l d -> b d l')
        

        y = self.selective_scan_cuda(x,dt,A,B,C,D,
                                     delta_softplus=True,
                                     delta_bias=self.dt_proj.bias.float())
        
        y = rearrange(z,'b d l -> b l d') # B x L x dim

        return y



class VisionMambaBlock(nn.Module):
    def __init__(self,
                 dim = 256,
                 dt_rank = 32,
                 dim_inner = 512,
                 d_state = 256):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

        # creating a “memory” of the previous states (shift-ssm)
        self.forward_conv1d = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=1)
        self.backward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()


        self.forward_ssm = SSM(dim,dt_rank, dim_inner, d_state)
        self.backward_ssm = SSM(dim,dt_rank, dim_inner, d_state)

        self.out_proj = nn.Linear(dim_inner,d_state,bias=False)
    
    def forward(self,x: torch.tensor):
        r'''
        Args:
            x (token sequence): B x L x d_state

        '''

        # save for residual connection
        skip_x = x
        
        # normalize input sequence
        x = self.norm(x)
        z = self.proj(x) # B x L x d_state
        x = self.proj(x) # B x L x d_state

        # process forward and backward
        ## Forward Conv1d -> forward SSM
        x1 = self.forward_conv1d(rearrange(x,'b l d -> b d l'))
        x1 = rearrange(x1,'b d l -> b l d')
        x1 = self.silu(x1)
        y_f = self.forward_ssm(x1,z)
        
        ## Backward
        x_b = x.flip([1]) # B x L x d_state
        z_b = z.flip([1])
        x2 = self.backward_conv1d(rearrange(x_b,'b l d -> b d l'))
        x2 = rearrange(x2,'b d l -> b l d')
        x2 = self.silu(x2)
        y_b = self.backward_ssm(x2,z_b)
        
        y = self.out_proj(y_f + y_b.flip([1]))
        # residual connection
        y = y + skip_x

        return y 

class VisionMamba(nn.Module):
    def __init__(self,
               dim = 256,
               dt_rank = 32,
               dim_inner = 512,
               d_state = 256,
               layer = 12):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size=16, stride=16)

        self.layers = nn.ModuleList([
            VisionMambaBlock(dim=dim,dt_rank=dt_rank,dim_inner=dim_inner,d_state=d_state)
            for _ in range(layer)
        ])

    def forward(self,x):
        '''
        x: batch_size, 3, 224, 224
        '''

        x = self.patch_embed(x) # B L D
        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self,x,caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        
        x = self.patch_embed(x) # B L D
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])









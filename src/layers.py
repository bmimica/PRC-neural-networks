import torch
import torch.nn as nn

import torch.nn.functional as F

# chooses automatically if do gpu or cpu calculations
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# to do a checkpoint
from torch.utils.checkpoint import checkpoint as cp
save_memory = False

# n_gene = fan_in; n_feature = fan_out
class multi_attention(nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, mode):
        # initializes parent class -> torch.nn.Module
        super(multi_attention, self).__init__()
        
        self.n_head = n_head # trains not only one, but a couple of "heads"
        self.n_gene = n_gene # this is the input size 
        self.batch_size = batch_size # batch, as in how many inputs you train at one training step
        self.n_feature = n_feature # as in number of outputs.
        self.mode = mode

        self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
        self.WK = nn.Parameter(torch.Tensor(self.n_head,n_feature,1), requires_grad=True)
        self.WV = nn.Parameter(torch.Tensor(self.n_head,n_feature,1), requires_grad=True)

        # xavier_normal_ helps for stability among layers: makes sure that data behavior doesn't explode. It "chooses" initial random numbers
        nn.init.xavier_normal_(self.WQ, gain=1)
        nn.init.xavier_normal_(self.WK, gain=1)
        nn.init.xavier_normal_(self.WV)

        """
        ask Davi about this W0 parameter
        """
        self.W_0=nn.Parameter(torch.Tensor(self.n_head*[0.001]), requires_grad=True)

    """
    ask about this to Davi: why not use dot product as in the paper? is eq 2
    """
    def QK_diff(self, Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq),2)
        return torch.nn.Softmax(dim=2)(QK_dif)
    

    """ 
    x = a data input of shape (batch_size, n_genes)

    """
    def attention(self, x):

        WQ = self.WQ
        WK = self.WK
        WV = self.WV 
        n_gene = self.n_gene

        # dim = (batch_size, 1, n_gene, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        # dim = (1, n_head, n_gene, 1)
        WQ = WQ.unsqueeze(0) 
        WK = WK.unsqueeze(0)
        WV = WV.unsqueeze(0) 

        # dim = (batch_size, n_head, n_gene, 1)
        Q = WQ * x 
        Q = torch.einsum('ijk,ij', WQ, x)
        K = WK * x
        V = WV * x

        Q = Q.expand(-1, -1, -1, n_gene) # copied into additional dimension
        K = K.expand(-1, -1, -1, n_gene).permute(0, 1, 3, 2) # same, but last dimensions are permuted: transposition

        # (QK)_bhgg' = Q_bhg K_bhg' = (WQ_hg * x_bg) * (WQ_hg' * x_bg')  
        # dim = (batch_size, n_head, n_gene, n_gene)
        QK = Q * K 
        z = torch.softmax(QK, dim=-1) # softmax over last dimensions, thats it over g' -> the Key genes

        def mask(x):
            mask = 1 - torch.eye(n_gene, device = device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            return x*mask
        
        # z @ V = (batch_size, n_head, n_gene, n_gene)@(batch_size, n_head, n_gene, 1) = (batch_size, n_head, n_gene, 1)
        # it sums like a_bhg = z_bhgg' V_bhg'
        z = mask(z)
        attention = z @ V

        if self.mode == 1:
            NotImplementedError("need to check this")

        return attention.squeeze(-1) # output = (batch_size, n_head, n_gene)
    

    """ 
    x = a data input of shape (batch_size, n_genes)
    output = has a size (batch_size, n_gene, 1)

    """
    def forward(self, x):
        if save_memory:
            a = cp(self.attention, x)
        else:
            a = self.attention(x)

        a_ = a.permute(0, 2, 1)
        output = a_ @ self.W_0 # colapses n_head results: dim = (batch_size, n_gene)
        return output
    

"""
Explain this
"""
class layernorm(nn.Module):
    def __init__(self, n_feature, eps=1e-6):
        super(layernorm, self).__init__()
        # a and b are learneable parameters. 
        self.a = nn.Parameter(torch.ones(n_feature))
        self.b = nn.Parameter(torch.zeros(n_feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
    
"""
x = input (batch_size, n_gene)
"""
class AttentionBlock(nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, mode, dropout = 0.1):
        super().__init__()
        self.attn = multi_attention(batch_size, n_head, n_gene, n_feature, mode)
        self.norm = layernorm(n_feature)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return x + self.norm(self.dropout(self.attn(x))) # output dim = x dim
    

class echo_state(nn.Module):
    def __init__(self, batch_size, n_head, fan_in, fan_out, R_size, sparsity = 0.95, spectral_radius = 0.9, leak_rate = 0.3):
        super().__init__()
        self.batch_size = batch_size
        self.n_head = n_head
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.R_size = R_size
        
        self.leak_rate = leak_rate

        self.W_in = nn.ParameterList(torch.empty(n_head, R_size, fan_out))
        nn.init.xavier_uniform_(self.W_in)

        self.W_out = nn.ParameterList()

        W_res_stack = []
        for _ in range(n_head):
            W = torch.randn(R_size, R_size)
            mask = (torch.rand_like(W) < sparsity).float()
            W = W * mask
            
            eigs = torch.linalg.eigvals(W)
            radius = torch.max(eigs.abs()).item()
            W = (W / (radius + 1e-9)) * spectral_radius 
            W_res_stack.append(W)

        self.register_buffer("W_res", torch.stack(W_res_stack))

    """
    again, x = (batch_size, fan_in)
    """
    def forward(self, x, act_function = torch.tanh):
        size = x.shape[0]
        state = torch.zeros(size, self.R_size)
        
        # F.linear(state, self.W_res) = state @ weight.T = (size, R_size) x (n_head, R_size, R_size) x (size, R_size) 
        preactivation = F.linear(state, self.W_res) + F.linear(x, self.W_in)
        new_state = act_function(preactivation)
        new_state = (1 - self.leak_rate)*state + self.leak_rate * new_state
        self.W_out.append(new_state)

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

        self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature), requires_grad=True)
        self.WK = nn.Parameter(torch.Tensor(self.n_head,n_feature), requires_grad=True)
        self.WV = nn.Parameter(torch.Tensor(self.n_head,n_feature), requires_grad=True)

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

        # dim = n_head, n_feature
        WQ = self.WQ
        WK = self.WK
        WV = self.WV 
        n_gene = self.n_gene

        # we will assume n_feature = n_gene for now
        """
        ask Davi about diff case: when n_feature is different from n_gene
        """
        Q = torch.einsum('bg,hg->bhg', x, WQ)
        K = torch.einsum('bg,hg->bhg', x, WK)
        V = torch.einsum('bg,hg->bhg', x, WV)
        # results in dim = (batch_size, n_head, n_gene)

        # (QK)_bhgg' = Q_bhg K_bhg' = (WQ_hg * x_bg) * (WQ_hg' * x_bg')  
        QK = torch.einsum('bhg, bhi -> bhgi', Q, K) 
        # results in dim = (batch_size, n_head, n_gene, n_gene)
        a = torch.softmax(QK, dim=-1) # softmax over last dimensions, thats it over g' -> the Key genes
        mask = 1 - torch.eye(n_gene, device = device)
        
        z = torch.einsum('bhgi, bhi -> bhg', a*mask , V)
        # dim = (batch_size, n_head, n_gene)

        if self.mode == 1:
            NotImplementedError("need to check this")

        return z # output = (batch_size, n_head, n_gene)
    

    """ 
    x = a data input of shape (batch_size, n_genes)
    output = has a size (batch_size, n_gene, 1)

    """
    def forward(self, x):
        if save_memory:
            z = cp(self.attention, x)
        else:
            z = self.attention(x)
        out = torch.einsum('bhg, h -> bg' , z, self.W_0) # colapses n_head results: dim = (batch_size, n_gene)
        return out
    

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
    def __init__(self, batch_size, n_head, fan_in, fan_out, R_size, sparsity = 0.95, spectral_radius = 0.9, leak_rate = 0.3, dropout = 0.3):
        super().__init__()
        self.batch_size = batch_size
        self.n_head = n_head
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.R_size = R_size
        
        self.leak_rate = leak_rate
        self.dropout = dropout

        self.W_in = nn.Parameter(torch.empty(n_head, R_size, fan_in), requires_grad=False)
        nn.init.xavier_uniform_(self.W_in)

        self.W_out = nn.Parameter(torch.empty(n_head, fan_out, R_size))
        nn.init.xavier_uniform_(self.W_out)

        W_res_stack = []
        for _ in range(n_head):
            W = torch.randn(R_size, R_size)
            mask = (torch.rand_like(W) < sparsity).float()
            W = W * mask
            
            eigs = torch.linalg.eigvals(W)
            radius = torch.max(eigs.abs()).item()
            W = (W / (radius + 1e-9)) * spectral_radius 
            W_res_stack.append(W)

        self.register_buffer("W_res", torch.stack(W_res_stack)) # dim = (n_head, R_size, R_size)
        self.collected_states = []

    """
    again, x = (batch_size, fan_in)
    """
    def forward(self, x, act_function = torch.tanh):
        W_res = self.W_res
        batch_size = self.batch_size
        state = torch.zeros(batch_size, self.n_head, self.R_size)

        # state (b, h, R_in) * W_res (h, R_out, R_in) -> (b, h, R_out)
        # x (b, f) * W_in (h, R, f) -> (b, h, R)
        preactivation = torch.einsum('bhi, hoi -> bho', state, W_res) + torch.einsum('bi, hRi -> bhR', x, self.W_in)
        new_state = act_function(preactivation)
        new_state = (1 - self.leak_rate)*state + self.leak_rate * new_state

        y = torch.einsum('bhr, hgr -> bhg', new_state, self.W_out)
        ''' ask davi : in attention we use a learning collapse, here just a mean
        '''
        norm = nn.BatchNorm1d(self.fan_out)
        out = norm(y)

        do = nn.Dropout(self.dropout)
        out = do(out)
        return out.mean(dim=1) # output = (batch_size, n_genes)
"""Minimal research baselines (educational implementations, not paper-exact reproductions)."""
import torch
from torch import nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    """FM + MLP on categorical user ID, user demographics, item ID, movie genres.
    Uses mean-pooled genre embeddings so multi-genre items are supported.
    """
    def __init__(self,n_users,n_items,n_age,n_gender,n_job,n_genres,d=32):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(n,d) for n in [n_users,n_items,n_age,n_gender,n_job,n_genres]])
        self.lin=nn.ModuleList([nn.Embedding(n,1) for n in [n_users,n_items,n_age,n_gender,n_job,n_genres]])
        self.mlp=nn.Sequential(nn.Linear(6*d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.bias=nn.Parameter(torch.zeros(()))
        # Small init: with N(0,1) embeddings the FM pairwise term starts with huge logits.
        for e in [*self.embs,*self.lin]: nn.init.normal_(e.weight,std=.01)
    def forward(self,u,i,age,gender,job,genre):
        ids=[u,i,age,gender,job]
        e=[emb(x) for emb,x in zip(self.embs[:5],ids)]
        l=[emb(x).squeeze(-1) for emb,x in zip(self.lin[:5],ids)]
        counts=genre.sum(dim=1,keepdim=True).clamp_min(1.)
        g=genre@self.embs[5].weight/counts
        gl=(genre@self.lin[5].weight).squeeze(-1)/counts.squeeze(-1)
        e.append(g);l.append(gl)
        stack=torch.stack(e,dim=1)
        fm=0.5*((stack.sum(1)**2-(stack**2).sum(1)).sum(-1))
        return fm+torch.stack(l,dim=1).sum(1)+self.mlp(torch.cat(e,dim=1)).squeeze(-1)+self.bias

class FieldDeepFM(nn.Module):
    """Dataset-agnostic DeepFM: user ID, item ID, any number of categorical user/item
    fields, and one mean-pooled multi-hot item field (genres, product categories, ...).
    Same FM + linear + MLP structure as DeepFM above.
    """
    def __init__(self,n_users,n_items,user_cards,item_cards,n_multi,d=32,init_std=None):
        super().__init__()
        cards=[n_users,n_items,*user_cards,*item_cards]
        self.n_user_fields=len(user_cards)
        self.embs=nn.ModuleList([nn.Embedding(n,d) for n in cards])
        self.lin=nn.ModuleList([nn.Embedding(n,1) for n in cards])
        self.multi=nn.Embedding(max(n_multi,1),d);self.multi_lin=nn.Embedding(max(n_multi,1),1)
        n_fields=len(cards)+1
        self.mlp=nn.Sequential(nn.Linear(n_fields*d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.bias=nn.Parameter(torch.zeros(()))
        if init_std:
            # N(0,1) embeddings make the FM pairwise term huge at start (initial BCE ~4.7).
            for emb in [*self.embs,*self.lin,self.multi,self.multi_lin]:nn.init.normal_(emb.weight,std=init_std)
    def forward(self,u,i,user_fields,item_fields,multi):
        ids=[u,i,*user_fields.unbind(-1),*item_fields.unbind(-1)]
        e=[emb(x) for emb,x in zip(self.embs,ids)]
        l=[emb(x).squeeze(-1) for emb,x in zip(self.lin,ids)]
        counts=multi.sum(dim=1,keepdim=True).clamp_min(1.)
        e.append(multi@self.multi.weight/counts)
        l.append((multi@self.multi_lin.weight).squeeze(-1)/counts.squeeze(-1))
        stack=torch.stack(e,dim=1)
        fm=0.5*((stack.sum(1)**2-(stack**2).sum(1)).sum(-1))
        return fm+torch.stack(l,dim=1).sum(1)+self.mlp(torch.cat(e,dim=1)).squeeze(-1)+self.bias

class SASRec(nn.Module):
    """Causal Transformer, next-item prediction. Item IDs shifted +1; 0 is padding."""
    def __init__(self,n_items,d=64,max_len=50,heads=2,layers=2,dropout=.1):
        super().__init__()
        self.max_len=max_len;self.heads=heads
        self.item=nn.Embedding(n_items+1,d,padding_idx=0)
        self.pos=nn.Embedding(max_len,d)
        layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=4*d,dropout=dropout,batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,layers)
        self.norm=nn.LayerNorm(d)
        nn.init.normal_(self.item.weight,std=.02);nn.init.normal_(self.pos.weight,std=.02)
        with torch.no_grad(): self.item.weight[0].zero_()
    def encode(self,seq):
        b,l=seq.shape
        x=self.item(seq)+self.pos(torch.arange(l,device=seq.device))[None,:,:]
        # Block future positions and padding keys, but always let a position see
        # itself. Otherwise left-padded rows are fully masked and attention
        # returns NaN in eval mode for every user with history < max_len.
        causal=torch.triu(torch.ones(l,l,device=seq.device,dtype=torch.bool),diagonal=1)
        eye=torch.eye(l,device=seq.device,dtype=torch.bool)
        blocked=(causal[None]|seq.eq(0)[:,None,:])&~eye[None]
        hidden=self.encoder(x,mask=blocked.repeat_interleave(self.heads,dim=0))
        return self.norm(hidden[:,-1,:])
    def forward(self,seq):
        return self.encode(seq)@self.item.weight[1:].T

class NGCF(nn.Module):
    """NGCF-style message passing with bi-interaction term and BPR loss.
    Adjacency normalized symmetrically; no external graph dependency.
    """
    def __init__(self,n_users,n_items,edges,d=32,layers=2):
        super().__init__()
        self.n_users=n_users;self.n_items=n_items
        self.user=nn.Embedding(n_users,d);self.item=nn.Embedding(n_items,d)
        nn.init.xavier_uniform_(self.user.weight);nn.init.xavier_uniform_(self.item.weight)
        self.linear=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        self.bi=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        self.register_buffer('adj',self.normalized_adj(edges))
    def normalized_adj(self,edges):
        u=torch.as_tensor(edges[:,0],dtype=torch.long)
        i=torch.as_tensor(edges[:,1],dtype=torch.long)+self.n_users
        src=torch.cat([u,i]);dst=torch.cat([i,u]);n=self.n_users+self.n_items
        degree=torch.bincount(src,minlength=n).float().clamp_min(1)
        values=degree[src].rsqrt()*degree[dst].rsqrt()
        return torch.sparse_coo_tensor(torch.stack([src,dst]),values,(n,n)).coalesce()
    def set_graph(self,edges):
        """Swap the propagation graph (e.g. add held-out history edges at inference)."""
        self.adj=self.normalized_adj(edges).to(self.user.weight.device)
    def encode(self):
        x=torch.cat([self.user.weight,self.item.weight]);layers=[x]
        for w1,w2 in zip(self.linear,self.bi):
            neighbor=torch.sparse.mm(self.adj,x)
            x=F.leaky_relu(w1(neighbor)+w2(x*neighbor),negative_slope=.2)
            layers.append(F.normalize(x,p=2,dim=-1))
        z=torch.cat(layers,dim=-1)
        return z[:self.n_users],z[self.n_users:]
    def bpr(self,u,i,j):
        users,items=self.encode()
        pos=(users[u]*items[i]).sum(-1)
        neg=(users[u]*items[j]).sum(-1)
        reg=1e-5*(self.user(u).square().sum()+self.item(i).square().sum()+self.item(j).square().sum())/len(u)
        return F.softplus(neg-pos).mean()+reg

import numpy as np
import torch
from src.models import SASRec
from src.fusion import fuse, Gate


def test_sasrec_no_nan_with_left_padding():
    torch.manual_seed(0)
    m=SASRec(20,d=16,max_len=6,heads=2);m.eval()
    seq=torch.tensor([[0,0,0,0,3,4],[1,2,3,4,5,6]])
    with torch.no_grad(): out=m(seq)
    assert torch.isfinite(out).all()


def test_sasrec_ignores_padding_tokens():
    torch.manual_seed(0)
    m=SASRec(20,d=16,max_len=6,heads=2);m.eval()
    with torch.no_grad():
        a=m(torch.tensor([[0,0,0,0,3,4]]))
        m.pos.weight[:4]=torch.randn(4,16)  # changing padded positions' embeddings must not matter
        b=m(torch.tensor([[0,0,0,0,3,4]]))
    assert torch.allclose(a,b,atol=1e-5)


def test_fuse_weights():
    z=np.random.default_rng(0).normal(size=(2,3,5)).astype(np.float32)
    assert np.allclose(fuse([1,0,0],z),z[:,0])
    w=np.array([[0,1,0],[.5,.5,0]],dtype=np.float32)
    assert np.allclose(fuse(w,z)[1],.5*z[1,0]+.5*z[1,1])
    g=Gate(4);x=torch.randn(7,4)
    assert torch.allclose(g(x).sum(-1),torch.ones(7))

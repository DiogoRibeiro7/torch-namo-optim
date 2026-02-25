import torch
from torch_namo import NAMO, NAMOD, orth

def test_smoke():
    m = torch.randn(8, 4)
    o = orth(m, method="newton_schulz", ns_iters=3)
    assert o.shape == m.shape

    p = torch.nn.Parameter(torch.randn(8, 4))
    _ = NAMO([p], lr=1e-3)
    _ = NAMOD([p], lr=1e-3, c=1.0)

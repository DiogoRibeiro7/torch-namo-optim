import pytest
import torch
from torch_namo import NAMO, NAMOD, orth


def test_smoke():
    m = torch.randn(8, 4)
    o = orth(m, method="newton_schulz", ns_iters=3)
    assert o.shape == m.shape

    p = torch.nn.Parameter(torch.randn(8, 4))
    _ = NAMO([p], lr=1e-3)
    _ = NAMOD([p], lr=1e-3, c=1.0)


def test_mu1_leq_mu2_validation():
    p = torch.nn.Parameter(torch.randn(8, 4))

    with pytest.raises(ValueError, match="mu1 must be <= mu2"):
        NAMO([p], lr=1e-3, mu1=0.99, mu2=0.95)

    with pytest.raises(ValueError, match="mu1 must be <= mu2"):
        NAMOD([p], lr=1e-3, mu1=0.99, mu2=0.95, c=1.0)

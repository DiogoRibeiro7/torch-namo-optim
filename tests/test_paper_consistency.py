import torch

from torch_namo import NAMO, NAMOD, orth


def test_namo_one_step_matches_formula():
    torch.manual_seed(0)
    p0 = torch.randn(5, 3)
    g = torch.randn(5, 3)
    p = torch.nn.Parameter(p0.clone())
    p.grad = g.clone()

    lr = 1e-3
    mu1 = 0.95
    mu2 = 0.99
    eps = 1e-8
    wd = 0.01

    opt = NAMO(
        [p],
        lr=lr,
        mu1=mu1,
        mu2=mu2,
        eps=eps,
        weight_decay=wd,
        orth_method="svd",
    )
    opt.step()

    m = (1.0 - mu1) * g
    v = (1.0 - mu2) * torch.sum(g * g)
    o = orth(m, method="svd")
    alpha = (torch.sqrt(torch.tensor(1.0 - mu2)) / (1.0 - mu1)) * (
        torch.linalg.norm(m, ord="fro") / (torch.sqrt(v) + eps)
    )
    expected = p0 - lr * alpha * (o + wd * p0)

    assert torch.allclose(p.detach(), expected, atol=1e-6, rtol=1e-6)


def test_namod_one_step_matches_formula():
    torch.manual_seed(1)
    p0 = torch.randn(6, 4)
    g = torch.randn(6, 4)
    p = torch.nn.Parameter(p0.clone())
    p.grad = g.clone()

    lr = 3e-4
    mu1 = 0.9
    mu2 = 0.99
    eps = 1e-8
    c = 0.3
    wd = 0.01

    opt = NAMOD(
        [p],
        lr=lr,
        mu1=mu1,
        mu2=mu2,
        eps=eps,
        c=c,
        weight_decay=wd,
        orth_method="svd",
    )
    opt.step()

    m = (1.0 - mu1) * g
    g_col_sq = torch.sum(g * g, dim=0)
    v = (1.0 - mu2) * g_col_sq
    o = orth(m, method="svd")

    numer = torch.sqrt(torch.tensor(1.0 - mu2))
    denom = 1.0 - mu1
    m_col = torch.sqrt(torch.sum(m * m, dim=0))
    d = (numer / denom) * (m_col / (torch.sqrt(v) + eps))
    dbar = torch.mean(torch.abs(d))
    d_tilde = torch.clamp(d, min=float(c * dbar), max=float(dbar / c))

    expected = p0 - lr * ((o + wd * p0) * d_tilde.view(1, -1))
    assert torch.allclose(p.detach(), expected, atol=1e-6, rtol=1e-6)

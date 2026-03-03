import torch

from torch_namo import NAMO, NAMOD, RoutedOptimizer, split_params


def _set_grad(p: torch.nn.Parameter, seed: int) -> None:
    gen = torch.Generator(device=p.device).manual_seed(seed)
    g = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
    p.grad = g


def test_namo_multi_step_updates_are_finite():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = NAMO([p], lr=1e-3, orth_method="svd")
    start = p.detach().clone()

    for step in range(5):
        _set_grad(p, seed=100 + step)
        opt.step()

    assert torch.isfinite(p).all()
    assert not torch.allclose(p.detach(), start)


def test_namod_state_dict_roundtrip_produces_same_next_step():
    torch.manual_seed(11)
    p1 = torch.nn.Parameter(torch.randn(6, 3))
    p2 = torch.nn.Parameter(p1.detach().clone())

    opt1 = NAMOD([p1], lr=2e-4, c=0.5, orth_method="svd")
    opt2 = NAMOD([p2], lr=2e-4, c=0.5, orth_method="svd")

    _set_grad(p1, seed=200)
    opt1.step()
    opt2.load_state_dict(opt1.state_dict())
    p2.data.copy_(p1.data)

    _set_grad(p1, seed=201)
    _set_grad(p2, seed=201)
    opt1.step()
    opt2.step()

    max_abs_err = torch.max(torch.abs(p1.detach() - p2.detach())).item()
    assert max_abs_err < 2e-4


def test_routed_optimizer_scaler_style_step_path():
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 3)
            self.norm = torch.nn.LayerNorm(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.lin(x)).sum()

    class DummyScaler:
        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def step(self, opt: torch.optim.Optimizer) -> None:
            opt.step()

        def update(self) -> None:
            pass

    torch.manual_seed(13)
    model = TinyModel()
    matrix_params, other_params = split_params(model)
    opt = RoutedOptimizer(
        matrix_params=matrix_params,
        other_params=other_params,
        matrix_opt_cls=NAMOD,
        matrix_opt_kwargs={"lr": 1e-3, "c": 1.0, "orth_method": "svd"},
        other_opt_cls=torch.optim.AdamW,
        other_opt_kwargs={"lr": 1e-3},
    )
    scaler = DummyScaler()

    x = torch.randn(5, 4)
    opt.zero_grad(set_to_none=True)
    loss = model(x)
    scaler.scale(loss).backward()
    scaler.step(opt.matrix_opt)
    scaler.step(opt.other_opt)
    scaler.update()

    params = [p.detach() for p in model.parameters()]
    assert all(torch.isfinite(p).all() for p in params)

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import math
import torch
from torch import Tensor
from torch.optim import Optimizer

OrthMethod = Literal["newton_schulz", "svd"]


@torch.no_grad()
def _orth_svd(m: Tensor) -> Tensor:
    """Exact Orth(M) via reduced SVD: M = U S V^T -> Orth(M) = U V^T."""
    if m.ndim != 2:
        raise ValueError(f"orth_svd expects 2D tensor, got shape {tuple(m.shape)}")
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    return u @ vh


@torch.no_grad()
def _invsqrt_sym_newton_schulz(a: Tensor, *, n_iters: int = 5, eps: float = 1e-12) -> Tensor:
    """
    Approximate A^{-1/2} for symmetric positive (semi-)definite A using Newton-Schulz.

    Implementation choices:
    - Symmetrize A to reduce numerical drift.
    - Normalize by trace for stability.
    - Fallback to identity if trace is too small.

    Args:
        a: Square matrix (k x k).
        n_iters: Number of iterations.
        eps: Small threshold for degeneracy checks.

    Returns:
        Approximate inverse square root of A.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got shape {tuple(a.shape)}")

    a_sym = 0.5 * (a + a.transpose(-1, -2))
    tr = torch.trace(a_sym)

    if (not torch.isfinite(tr)) or tr.abs().item() <= eps:
        return torch.eye(a.shape[0], device=a.device, dtype=a.dtype)

    a0 = a_sym / tr
    i = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)

    y = a0
    z = i

    for _ in range(int(n_iters)):
        zy = z @ y
        t = 0.5 * (3.0 * i - zy)
        y = y @ t
        z = t @ z

    return z / torch.sqrt(tr)


@torch.no_grad()
def orth(
    m: Tensor,
    *,
    method: OrthMethod = "newton_schulz",
    ns_iters: int = 5,
    ns_eps: float = 1e-12,
) -> Tensor:
    """Compute Orth(M) for a 2D tensor."""
    if m.ndim != 2:
        raise ValueError(f"orth expects 2D tensor, got shape {tuple(m.shape)}")

    if method == "svd":
        return _orth_svd(m)
    if method != "newton_schulz":
        raise ValueError(f"Unknown orth method: {method!r}")

    rows, cols = m.shape
    if cols <= rows:
        a = m.transpose(0, 1) @ m
        invsqrt = _invsqrt_sym_newton_schulz(a, n_iters=ns_iters, eps=ns_eps)
        return m @ invsqrt

    a = m @ m.transpose(0, 1)
    invsqrt = _invsqrt_sym_newton_schulz(a, n_iters=ns_iters, eps=ns_eps)
    return invsqrt @ m


def _is_matrix(p: Tensor) -> bool:
    """NAMO/NAMO-D are defined for matrix-shaped parameters (2D tensors)."""
    return p.ndim == 2


class NAMO(Optimizer):
    """
    NAMO optimizer for 2D parameters only.

    Summary:
      - Momentum: M_t = mu1 M_{t-1} + (1-mu1) G_t
      - Second moment (scalar): v_t = mu2 v_{t-1} + (1-mu2) ||G_t||_F^2
      - Direction: Orth(M_t)
      - Scale: alpha_t = sqrt(1-mu2^t)/(1-mu1^t) * ||M_t||_F / (sqrt(v_t)+eps)
      - Update: W <- W - lr * alpha_t * (Orth(M_t) + weight_decay * W)

    Production details:
      - `stable_dtype` (default float32) is used internally to keep AMP stable.
      - States live on the same device as parameters.
    """

    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        *,
        lr: float = 1e-3,
        mu1: float = 0.95,
        mu2: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        orth_method: OrthMethod = "newton_schulz",
        ns_iters: int = 5,
        ns_eps: float = 1e-12,
        stable_dtype: torch.dtype = torch.float32,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0.")
        if not (0.0 <= mu1 < 1.0 and 0.0 <= mu2 < 1.0):
            raise ValueError("mu1 and mu2 must be in [0,1).")
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0.")
        if ns_iters <= 0:
            raise ValueError("ns_iters must be >= 1.")
        if ns_eps <= 0.0:
            raise ValueError("ns_eps must be > 0.")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            mu1=mu1,
            mu2=mu2,
            eps=eps,
            weight_decay=weight_decay,
            orth_method=orth_method,
            ns_iters=ns_iters,
            ns_eps=ns_eps,
            stable_dtype=stable_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                out = closure()
            loss = float(out) if out is not None else None

        for group in self.param_groups:
            lr = float(group["lr"])
            mu1 = float(group["mu1"])
            mu2 = float(group["mu2"])
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            orth_method = cast(OrthMethod, group["orth_method"])
            ns_iters = int(group["ns_iters"])
            ns_eps = float(group["ns_eps"])
            stable_dtype = cast(torch.dtype, group["stable_dtype"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not _is_matrix(p):
                    raise ValueError(f"NAMO expects 2D params, got shape {tuple(p.shape)}")

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("NAMO does not support sparse gradients.")

                st = self.state[p]
                if len(st) == 0:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros((), device=p.device, dtype=stable_dtype)

                st["t"] += 1
                t = int(st["t"])

                m: Tensor = st["m"]
                m.mul_(mu1).add_(g, alpha=(1.0 - mu1))

                g_ = g.to(dtype=stable_dtype)
                g_frob_sq = torch.sum(g_ * g_)
                v: Tensor = st["v"]
                v.mul_(mu2).add_(g_frob_sq, alpha=(1.0 - mu2))

                m_ = m.to(dtype=stable_dtype)
                o = orth(m_, method=orth_method, ns_iters=ns_iters, ns_eps=ns_eps)

                mu1_t = mu1**t
                mu2_t = mu2**t
                numer = math.sqrt(max(0.0, 1.0 - mu2_t))
                denom = max(1e-20, 1.0 - mu1_t)

                m_frob = torch.sqrt(torch.sum(m_ * m_))
                alpha = (numer / denom) * float(m_frob / (torch.sqrt(v) + eps))

                update = o
                if wd > 0.0:
                    update = update + (wd * p.to(dtype=stable_dtype))

                p.add_(update.to(dtype=p.dtype), alpha=-lr * alpha)

        return loss


class NAMOD(Optimizer):
    """
    NAMO-D optimizer for 2D parameters only.

    Differences vs NAMO:
      - v_t is per-column (vector)
      - d_t is per-column stepsize vector
      - clamp d_t around mean(|d_t|) using c in (0,1]
      - apply right scaling (column-wise multiplication)
    """

    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        *,
        lr: float = 1e-3,
        mu1: float = 0.95,
        mu2: float = 0.99,
        eps: float = 1e-8,
        c: float = 1.0,
        weight_decay: float = 0.0,
        orth_method: OrthMethod = "newton_schulz",
        ns_iters: int = 5,
        ns_eps: float = 1e-12,
        stable_dtype: torch.dtype = torch.float32,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0.")
        if not (0.0 <= mu1 < 1.0 and 0.0 <= mu2 < 1.0):
            raise ValueError("mu1 and mu2 must be in [0,1).")
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        if not (0.0 < c <= 1.0):
            raise ValueError("c must be in (0,1].")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0.")
        if ns_iters <= 0:
            raise ValueError("ns_iters must be >= 1.")
        if ns_eps <= 0.0:
            raise ValueError("ns_eps must be > 0.")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            mu1=mu1,
            mu2=mu2,
            eps=eps,
            c=c,
            weight_decay=weight_decay,
            orth_method=orth_method,
            ns_iters=ns_iters,
            ns_eps=ns_eps,
            stable_dtype=stable_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                out = closure()
            loss = float(out) if out is not None else None

        for group in self.param_groups:
            lr = float(group["lr"])
            mu1 = float(group["mu1"])
            mu2 = float(group["mu2"])
            eps = float(group["eps"])
            c = float(group["c"])
            wd = float(group["weight_decay"])
            orth_method = cast(OrthMethod, group["orth_method"])
            ns_iters = int(group["ns_iters"])
            ns_eps = float(group["ns_eps"])
            stable_dtype = cast(torch.dtype, group["stable_dtype"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not _is_matrix(p):
                    raise ValueError(f"NAMOD expects 2D params, got shape {tuple(p.shape)}")

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("NAMOD does not support sparse gradients.")

                _, cols = p.shape

                st = self.state[p]
                if len(st) == 0:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros((cols,), device=p.device, dtype=stable_dtype)

                st["t"] += 1
                t = int(st["t"])

                m: Tensor = st["m"]
                m.mul_(mu1).add_(g, alpha=(1.0 - mu1))

                g_ = g.to(dtype=stable_dtype)
                g_col = torch.sqrt(torch.sum(g_ * g_, dim=0))
                v: Tensor = st["v"]
                v.mul_(mu2).add_(g_col * g_col, alpha=(1.0 - mu2))

                m_ = m.to(dtype=stable_dtype)
                o = orth(m_, method=orth_method, ns_iters=ns_iters, ns_eps=ns_eps)

                mu1_t = mu1**t
                mu2_t = mu2**t
                numer = math.sqrt(max(0.0, 1.0 - mu2_t))
                denom = max(1e-20, 1.0 - mu1_t)

                m_col = torch.sqrt(torch.sum(m_ * m_, dim=0))
                d = (numer / denom) * (m_col / (torch.sqrt(v) + eps))

                dbar = torch.mean(torch.abs(d))
                lo = c * dbar
                hi = dbar / c
                d_tilde = torch.clamp(d, min=float(lo), max=float(hi))

                scaled = o * d_tilde.view(1, -1)
                if wd > 0.0:
                    scaled = scaled + (wd * p.to(dtype=stable_dtype)) * d_tilde.view(1, -1)

                p.add_(scaled.to(dtype=p.dtype), alpha=-lr)

        return loss


class RoutedOptimizer:
    """
    Route parameters to:
      - NAMO/NAMO-D for 2D matrices
      - another optimizer (default AdamW) for everything else

    API:
      - step()
      - zero_grad()
      - state_dict() / load_state_dict()
      - exposes `matrix_opt` and `other_opt` for AMP scaler.step(...)
    """

    def __init__(
        self,
        *,
        matrix_params: Iterable[Tensor],
        other_params: Iterable[Tensor],
        matrix_opt_cls: type[Optimizer],
        matrix_opt_kwargs: Dict[str, Any],
        other_opt_cls: type[Optimizer] = torch.optim.AdamW,
        other_opt_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.matrix_params: List[Tensor] = list(matrix_params)
        self.other_params: List[Tensor] = list(other_params)
        other_opt_kwargs = {} if other_opt_kwargs is None else dict(other_opt_kwargs)

        self.matrix_opt: Optimizer = matrix_opt_cls(self.matrix_params, **matrix_opt_kwargs)
        self.other_opt: Optimizer = other_opt_cls(self.other_params, **other_opt_kwargs)

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return list(self.matrix_opt.param_groups) + list(self.other_opt.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.matrix_opt.zero_grad(set_to_none=set_to_none)
        self.other_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                out = closure()
            loss = float(out) if out is not None else None

        self.matrix_opt.step(None)
        self.other_opt.step(None)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {"matrix_opt": self.matrix_opt.state_dict(), "other_opt": self.other_opt.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.matrix_opt.load_state_dict(state_dict["matrix_opt"])
        self.other_opt.load_state_dict(state_dict["other_opt"])


def split_params(model: torch.nn.Module) -> Tuple[List[Tensor], List[Tensor]]:
    """Split model parameters into (2D matrices, others)."""
    matrix_params: List[Tensor] = []
    other_params: List[Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            matrix_params.append(p)
        else:
            other_params.append(p)
    return matrix_params, other_params

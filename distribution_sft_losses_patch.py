# Copyright 2025 the distribution_sft authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""P3D: Uniform and Capped targets with online top-K soft cross-entropy.

Copy this complete file to:
    src/llamafactory/train/distribution_sft_losses.py

The target is built from detached current-student logits at demonstration
prefixes. Both target construction and soft CE use a renormalized top-K
support containing the demonstrated token. No separate teacher is needed.
The function follows Transformers' compute_loss_func contract.

The legacy ppd names and default delta=0.05 are retained for compatibility;
the supplied training examples explicitly select delta=0.30.
The Uniform implementation is preserved. Capped adds the general KL
projection, solved by log-space bisection, and fits its detached soft target.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _flatten_shift(
    outputs, labels: torch.Tensor, ignore_index: int = -100
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Standard causal-LM shift + flatten + valid-mask filter."""
    logits = outputs.get("logits")
    if logits is None:
        raise ValueError("P3D requires model outputs containing logits.")

    logits = logits.float()
    vocab_size = logits.size(-1)
    labels = F.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    valid_mask = shift_labels != ignore_index
    if not valid_mask.any():
        return None

    return logits[valid_mask], shift_labels[valid_mask]


def _reduce(per_token_loss: torch.Tensor, num_items_in_batch) -> torch.Tensor:
    if num_items_in_batch is not None:
        total = per_token_loss.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total.device)
        return total / num_items_in_batch
    return per_token_loss.mean()


def _topk_with_ystar(logits: torch.Tensor, labels: torch.Tensor, topk: int) -> torch.Tensor:
    """Return [N, K] indices such that y* is guaranteed to appear in each row.

    If y* is already in the natural top-K, return that. Otherwise replace the
    lowest-ranked top-K position with y* so PPD's target c* has a slot to
    write to. Stop-grad by construction (topk indices are not differentiable).
    """
    _, top_idx = logits.topk(k=topk, dim=-1)
    in_topk = top_idx.eq(labels.unsqueeze(-1)).any(-1)  # [N]
    if in_topk.all():
        return top_idx
    fixed = top_idx.clone()
    fixed[~in_topk, -1] = labels[~in_topk]
    return fixed


def ppd_online_loss_func(
    outputs,
    labels: torch.Tensor,
    num_items_in_batch=None,
    delta: float = 0.05,
    topk: int = 200,
    ignore_index: int = -100,
) -> torch.Tensor:
    r"""Peak-Preserving Prior Distribution, online, top-K soft-CE.

    All top-K work is renormalised on the top-K support (P_theta and Q* both
    normalised over the same top-K). y* is guaranteed to lie in the support.
    Gradient at logit level (over top-K) is P_theta^topk - Q*_topk.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1].")
    logits = outputs.get("logits")
    if logits is not None and not 2 <= topk <= logits.size(-1):
        raise ValueError("topk must be between 2 and the vocabulary size.")

    prepared = _flatten_shift(outputs, labels, ignore_index)
    if prepared is None:
        # Keep a gradient path when a whole batch contains no supervised tokens.
        return outputs["logits"].sum() * 0.0

    logits_v, labels_v = prepared

    with torch.no_grad():
        top_idx = _topk_with_ystar(logits_v.detach(), labels_v, topk)          # [N, K]

        top_z_sg = logits_v.detach().gather(-1, top_idx)                       # [N, K]
        log_p_topk_sg = F.log_softmax(top_z_sg, dim=-1)                        # [N, K]
        p_topk_sg = log_p_topk_sg.exp()                                        # [N, K]

        is_ystar = top_idx.eq(labels_v.unsqueeze(-1))                          # [N, K]
        p_star = (p_topk_sg * is_ystar.to(p_topk_sg.dtype)).sum(-1)            # [N]
        p_minus = p_topk_sg.masked_fill(is_ystar, 0.0).max(-1).values          # [N]

        denom = (1.0 - p_star).clamp_min(1e-6)
        c_flip = p_minus / (1.0 - p_star + p_minus).clamp_min(1e-6)
        c_star = torch.maximum(p_star, c_flip + delta).clamp(max=1.0 - 1e-6)
        eps = (1.0 - c_star) / denom                                           # [N]

        q_star_topk = eps.unsqueeze(-1) * p_topk_sg                            # [N, K]
        q_star_topk = torch.where(is_ystar, c_star.unsqueeze(-1), q_star_topk)

    top_z_grad = logits_v.gather(-1, top_idx)                                  # [N, K]  grad
    log_p_topk_grad = F.log_softmax(top_z_grad, dim=-1)                        # [N, K]  grad

    per_token = -(q_star_topk * log_p_topk_grad).sum(-1)                       # [N]
    return _reduce(per_token, num_items_in_batch)


@torch.no_grad()
def capped_projection(
    log_p: torch.Tensor,
    target: torch.Tensor,
    gap=0.0,
    iterations: int = 40,
) -> torch.Tensor:
    r"""Minimize KL(Q || P) subject to Q_y - Q_j >= gap for every j != y.

    Args:
        log_p: Finite, normalized log-probabilities, shape [N, K], K >= 2.
        target: Demonstrated-token indices within that support, shape [N].
        gap: Scalar or [N] tensor in [0, 1]. Zero gives the original
            weak-dominance projection; this is not the Uniform delta.
        iterations: Number of bisection iterations (40 in the experiments).

    The solution has Q_j = min(a * P_j, Q_y - gap) for non-target j.
    We solve the normalization and KKT conditions for the clipping threshold
    in log space. No sorting of clipping sets or repeated KL evaluation is
    needed. FP16/BF16 inputs are promoted to FP32; FP64 is retained.
    The returned target is detached from all inputs.
    """
    if log_p.ndim != 2 or log_p.shape[1] < 2 or not log_p.is_floating_point():
        raise ValueError("log_p must be a floating-point [N, K] tensor with K >= 2.")
    if target.shape != (log_p.shape[0],) or target.dtype not in (torch.int32, torch.int64):
        raise ValueError("target must contain one integer support index per row.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    if log_p.dtype != torch.float64:
        log_p = log_p.float()
    target = target.to(device=log_p.device, dtype=torch.long)
    gap = torch.as_tensor(gap, device=log_p.device, dtype=log_p.dtype)
    if gap.ndim == 0:
        gap = gap.expand(log_p.shape[0])
    if gap.shape != (log_p.shape[0],):
        raise ValueError("gap must be a scalar or have shape [N].")
    if not torch.isfinite(log_p).all():
        raise ValueError("log_p must be finite; use log_softmax on finite logits.")
    log_mass = torch.logsumexp(log_p, dim=-1)
    if not torch.allclose(log_mass, torch.zeros_like(log_mass), atol=1e-5, rtol=0):
        raise ValueError("log_p must be normalized on the selected support.")
    if ((target < 0) | (target >= log_p.shape[1])).any():
        raise ValueError("target contains an out-of-range support index.")
    if not torch.isfinite(gap).all() or ((gap < 0) | (gap > 1)).any():
        raise ValueError("gap must be finite and lie in [0, 1].")

    p = log_p.exp()
    is_target = F.one_hot(target, p.shape[1]).bool()
    p_y = p.gather(-1, target[:, None]).squeeze(-1)
    max_other = p.masked_fill(is_target, 0).amax(-1)
    q = p.clone()
    saturated = gap >= 1
    q[saturated] = is_target[saturated].to(q.dtype)
    active = (p_y - max_other < gap) & ~saturated
    if not active.any():
        return q

    lp = log_p[active]
    y = is_target[active]
    g = gap[active]
    lp_y = lp.gather(-1, target[active, None]).squeeze(-1)
    lr = lp.masked_fill(y, -torch.inf)
    n = p.shape[1] - 1
    log_one_minus_g = torch.log1p(-g)
    log_g = torch.log(g)  # -inf at zero is intentional.
    lo = torch.minimum(lp_y, lp.masked_fill(y, torch.inf).amin(-1))
    lo = lo + (log_one_minus_g - torch.log1p(n * g)) / (n + 1) - 1
    hi = lr.amax(-1)
    for _ in range(iterations):
        u = (lo + hi) * 0.5
        log_s = torch.logsumexp(torch.minimum(lr, u[:, None]), dim=-1)
        residual = (
            torch.logaddexp(u, log_g + log_s)
            - log_one_minus_g
            - lp_y
            - (lr - u[:, None]).clamp_min(0).sum(-1)
        )
        hi = torch.where(residual >= 0, u, hi)
        lo = torch.where(residual < 0, u, lo)

    u = (lo + hi) * 0.5
    capped_lr = torch.minimum(lr, u[:, None])
    log_s = torch.logsumexp(capped_lr, dim=-1)
    log_a = log_one_minus_g - torch.logaddexp(u, log_s)
    qa = (log_a[:, None] + capped_lr).exp()
    qa = torch.where(y, (g + (log_a + u).exp())[:, None], qa)
    q[active] = qa
    return q


def _capped_log_probs(outputs, labels, topk, ignore_index):
    """Select support in FP32 chunks without materializing all valid N x V logits."""
    logits = outputs["logits"]
    flat = logits.reshape(-1, logits.shape[-1])
    shifted = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].reshape(-1).to(logits.device)
    rows = torch.nonzero(shifted != ignore_index, as_tuple=False).squeeze(-1)
    if rows.numel() == 0:
        return None
    targets = shifted.index_select(0, rows)
    with torch.no_grad():
        supports = []
        for start in range(0, rows.numel(), 128):
            work = flat.detach().index_select(0, rows[start : start + 128]).float()
            supports.append(_topk_with_ystar(work, targets[start : start + 128], topk))
        support = torch.cat(supports, dim=0)
        target = support.eq(targets[:, None]).long().argmax(-1)
    selected = flat[rows[:, None], support].float()
    return F.log_softmax(selected, dim=-1), target


def ppd_capped_online_loss_func(
    outputs,
    labels: torch.Tensor,
    num_items_in_batch=None,
    delta: float = 0.05,
    topk: int = 200,
    ignore_index: int = -100,
    iterations: int = 40,
) -> torch.Tensor:
    r"""Fit the general Capped target on target-inclusive, normalized top-K.

    delta=0 uses the original weak-dominance KL projection. For delta>0,
    use Uniform's probability-floor offset and numerical guards to obtain
    its actual leading gap, gamma = c - alpha * max_non_target(P).
    Capped then minimizes KL(Q || P) at that same gap. Thus delta=0.30
    does not mean a constant 0.30 leading gap at every position.

    The target is detached and the loss is its full cross-entropy, not a
    weighted-NLL shortcut. Causal shifting, ignored labels and reduction
    follow the same compute_loss_func contract as ppd_online_loss_func.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1].")
    logits = outputs.get("logits")
    if logits is None:
        raise ValueError("P3D requires model outputs containing logits.")
    if not isinstance(topk, int) or not 2 <= topk <= logits.shape[-1]:
        raise ValueError("topk must be between 2 and the vocabulary size.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    prepared = _capped_log_probs(outputs, labels, topk, ignore_index)
    if prepared is None:
        return logits.sum() * 0.0
    log_p, target = prepared
    with torch.no_grad():
        if delta == 0:
            gap = torch.zeros_like(log_p[:, 0])
        else:
            p = log_p.detach().exp()
            is_target = F.one_hot(target, p.shape[1]).bool()
            p_y = p.gather(-1, target[:, None]).squeeze(-1)
            m = p.masked_fill(is_target, 0).amax(-1)
            boundary = m / (1 - p_y + m).clamp_min(1e-6)
            c = torch.maximum(p_y, boundary + delta).clamp(max=1 - 1e-6)
            alpha = (1 - c) / (1 - p_y).clamp_min(1e-6)
            gap = (c - alpha * m).clamp(0, 1)
        q = capped_projection(log_p.detach(), target, gap, iterations)
    return _reduce(-(q * log_p).sum(-1), num_items_in_batch)

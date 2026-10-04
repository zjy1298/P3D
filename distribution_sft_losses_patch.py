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
"""P3D: Peak-Preserving Prior Distribution, online top-K soft cross-entropy.

Copy this complete file to:
    src/llamafactory/train/distribution_sft_losses.py

The target is built from detached current-student logits at demonstration
prefixes. Both target construction and soft CE use a renormalized top-K
support containing the demonstrated token. No separate teacher is needed.
The function follows Transformers' compute_loss_func contract.

The legacy ppd names and default delta=0.05 are retained for compatibility;
the supplied training examples explicitly select delta=0.30.
Release changes: documentation, argument validation, and a differentiable
zero for fully masked batches. The experimental target formula is unchanged.
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

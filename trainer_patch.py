# Licensed under the Apache License, Version 2.0. See the license text in README.md.
# Target: src/llamafactory/train/sft/trainer.py
# Paste the block below INSIDE CustomSeq2SeqTrainer.__init__, using eight-space
# indentation, immediately BEFORE `if finetuning_args.use_dft_loss:`.
# It runs after super().__init__. Do not replace the file or compute_loss().
# This file is a paste-in snippet, not a standalone program.

if finetuning_args.use_ppd_loss:
    from functools import partial as _p3d_partial

    from ..distribution_sft_losses import ppd_online_loss_func

    if any(
        getattr(finetuning_args, name, False)
        for name in ("use_dft_loss", "use_eaft_loss", "use_asft_loss")
    ):
        raise ValueError("Enable only one training loss at a time.")
    if self.args.label_smoothing_factor != 0.0:
        raise ValueError("Set label_smoothing_factor=0.0 when using P3D.")
    if not 0.0 <= finetuning_args.ppd_delta <= 1.0:
        raise ValueError("ppd_delta must be in [0, 1].")
    if finetuning_args.loss_topk < 2:
        raise ValueError("loss_topk must be at least 2.")
    if model_args is not None and getattr(model_args, "enable_liger_kernel", False):
        raise ValueError("Disable enable_liger_kernel: P3D needs the model logits.")

    self.compute_loss_func = _p3d_partial(
        ppd_online_loss_func,
        delta=finetuning_args.ppd_delta,
        topk=finetuning_args.loss_topk,
    )

# Licensed under the Apache License, Version 2.0. See the license text in README.md.
# Target: src/llamafactory/hparams/finetuning_args.py
# Paste the fields below INSIDE the existing FinetuningArguments dataclass,
# immediately before `freeze_vision_tower`, using four-space indentation.
# `field` is already imported by the upstream file. Do not replace the file.
# This file is a paste-in snippet, not a standalone program.

use_ppd_loss: bool = field(
    default=False,
    metadata={"help": "Use P3D (Peak-Preserving Prior Distribution) SFT loss."},
)
ppd_delta: float = field(
    default=0.05,
    metadata={"help": "P3D probability-floor offset in [0, 1]. Paper examples use 0.30."},
)
loss_topk: int = field(
    default=200,
    metadata={"help": "P3D support size, including the demonstrated token; 2 <= K <= vocabulary size."},
)

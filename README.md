# P³D — Peak-Preserving Prior Distribution for RL-Ready Large Language Models

[Overview](#overview) · [Code](#code) · [Quick start](#quick-start) · [Parameters](#parameters) · [Examples](#examples)

<p align="center">
  <img src="method.png" alt="P³D overview: learning from demonstrations while retaining useful prior structure for subsequent reinforcement learning" width="100%">
</p>

<p align="center">
  <em>Balancing acquisition of demonstrated behavior with retention of useful alternatives.</em>
</p>

## Overview

Pretrained language models already encode broad knowledge and a rich prior over
plausible continuations. In the SFT-then-RL pipeline, supervised fine-tuning acts
as a bridge: it learns from expert demonstrations while preparing an
initialization from which reinforcement learning can explore and improve.
Effective SFT therefore needs to balance **acquisition** of demonstrated behavior
with **retention** of useful alternatives.

Standard one-hot supervision pushes probability toward the demonstrated token
without specifying which preferences among alternative continuations should
survive. This can narrow the exploration space available to subsequent RL.
Smoothing and regularization can reduce excessive concentration, but uncertainty
alone does not identify which alternatives are useful or preserve their relative
preferences.

We introduce **Peak-Preserving Prior Distribution (P³D)** to address this balance
through the design of the supervision target. Demonstrations specify what to
learn, while the student's own predictions provide the structure to retain.
P³D constructs a soft target that promotes the demonstrated token while preserving
relative preferences among non-target alternatives. This provides richer
supervision without requiring teacher probability vectors or a separate teacher
model during SFT, helping the model learn from demonstrations while retaining
useful foundations for subsequent reinforcement learning.

## Code

The release provides the P³D loss, two LLaMA-Factory integration snippets, and
SFT configurations. The code retains the `ppd` parameter names used in the
original implementation.

| File | Purpose |
| :--- | :--- |
| [distribution_sft_losses_patch.py](distribution_sft_losses_patch.py) | Complete P³D loss implementation |
| [finetuning_args_patch.py](finetuning_args_patch.py) | Three configurable loss parameters |
| [trainer_patch.py](trainer_patch.py) | Connect the loss to the SFT trainer |
| [configs/](configs/) | Mathematics, medical, and DeepSpeed configurations |
| [requirements.txt](requirements.txt) | Recorded SFT environment versions |

> **Release scope:** The code below covers the SFT implementation. Models,
> training datasets, GRPO, and the complete evaluation pipeline are not included.

## Quick start

### 1. Set up the environment

Use Linux and keep `P3D/` and `LLaMA-Factory/` in the same parent directory:

```text
workspace/
├── P3D/
└── LLaMA-Factory/
```

Run from that parent directory:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout ca50f22c38a77e72a4a21ef177ce4aa8f29d6930

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r ../P3D/requirements.txt
python -m pip install --no-deps -e .
```

| Python | PyTorch | CUDA | Attention |
| :---: | :---: | :---: | :---: |
| 3.12.12 | 2.10.0 | 12.8 | SDPA |

<details>
<summary>Environment notes</summary>

`requirements.txt` records the currently available SFT environment, not a
historical per-run lockfile. The `--no-deps` editable install preserves these
versions: the pinned upstream commit's package metadata caps Accelerate at
1.11.0, although its runtime check allows the recorded 1.13.0.
FlashAttention is optional; the supplied examples use PyTorch SDPA.

</details>

### 2. Apply the integration

All paths below are relative to `LLaMA-Factory/`. The `_patch.py` files are
**code to copy**, not standalone programs.

**Add the loss file.**

```bash
cp ../P3D/distribution_sft_losses_patch.py \
  src/llamafactory/train/distribution_sft_losses.py
```

**Add the parameter fields.** Open `src/llamafactory/hparams/finetuning_args.py`.
Paste [finetuning_args_patch.py](finetuning_args_patch.py) inside
`FinetuningArguments`, immediately before `freeze_vision_tower`, with **4-space
indentation**. The upstream file already imports `field`.

**Connect the trainer.** Open `src/llamafactory/train/sft/trainer.py`.
Paste [trainer_patch.py](trainer_patch.py) inside
`CustomSeq2SeqTrainer.__init__`, immediately before
`if finetuning_args.use_dft_loss:`, with **8-space indentation**.

Keep the existing trainer and dataclass. No change to `compute_loss()` is needed.

### 3. Register your data

Create `data/p3d/math_sft.json` and/or `data/p3d/medical_sft.json` using a JSON
array of prompt–response records:

```json
[
  {
    "instruction": "Your question",
    "input": "",
    "output": "Your reference answer"
  }
]
```

Then create `data/p3d/dataset_info.json`:

```json
{
  "math_sft": {"file_name": "math_sft.json"},
  "medical_sft": {"file_name": "medical_sft.json"}
}
```

Only the dataset selected for a run needs to exist. These examples describe the
input format; they do not reconstruct the paper's data mixture or preprocessing.

### 4. Start training

Run from `LLaMA-Factory/` after applying the snippets and preparing your data:

```bash
# Mathematics · Qwen3-4B · delta = 0.30
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/math.yaml

# Medical · Qwen3-4B · delta = 0.30
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/medical.yaml
```

The supplied configurations use **8 GPUs**, full fine-tuning, BF16, DeepSpeed
ZeRO-3, a sequence length of 4096, 3 epochs, and 60 warmup steps.
Use a fresh `output_dir` for each run.

## Parameters

### P³D loss

| Parameter | Code default | Description |
| :--- | :---: | :--- |
| `use_ppd_loss` | `false` | Enable P³D. Set to `false` for standard SFT with other custom losses and label smoothing disabled. |
| `ppd_delta` | `0.05` | Probability-floor offset in `[0, 1]`. The supplied configurations use **0.30**. |
| `loss_topk` | `200` | Support size, including the demonstrated token. Requires `2 <= K <= vocabulary size`. |

Example margins: `0.05`, `0.15`, **`0.30`**, `0.45`, `0.60`, and `0.80`.
The paper uses `loss_topk=200`; changing it changes the approximation.
Disable other custom losses, label smoothing, and fused Liger loss when using P³D.

<details>
<summary>How the margin and support affect the loss</summary>

Both the detached target and the fitted probabilities are normalized on the
same top-K support. If the demonstrated token is missing, it replaces the last
selected token. The original `1e-6` numerical guards are retained.

With `delta=0`, the local gradient can vanish when dominance already holds.
Increasing delta strengthens supervision. A large delta approaches a one-hot
target on the selected support; this differs from full-vocabulary CE.

</details>

### Training configuration

| Setting | Mathematics | Medical |
| :--- | :---: | :---: |
| Learning rate | `1e-5` | `1e-5` |
| Micro-batch per GPU | `8` | `8` |
| Gradient accumulation | `8` | `2` |
| Effective batch size on 8 GPUs | `512` | `128` |

Set `model_name_or_path` to a compatible Hugging Face model ID or your local
model directory, and match its template: `qwen3` or `llama3`.
Model access is subject to the upstream model terms.

## Examples

Override YAML settings with `key=value` arguments. Run all examples from
`LLaMA-Factory/`.

<details>
<summary><strong>Change the margin or support size</strong></summary>

```bash
# A larger probability-floor offset
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/math.yaml \
  ppd_delta=0.60 \
  output_dir=saves/p3d/math_d060

# A smaller support (optional variant)
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/math.yaml \
  loss_topk=100 \
  output_dir=saves/p3d/math_k100
```

</details>

<details>
<summary><strong>Run a standard SFT comparison</strong></summary>

```bash
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/math.yaml \
  use_ppd_loss=false \
  output_dir=saves/p3d/math_hardsft
```

</details>

<details>
<summary><strong>Use Qwen3-8B or Llama-3-8B</strong></summary>

Reduce the micro-batch size to `4` and increase accumulation to `16` to preserve
an effective batch size of 512 on 8 GPUs:

```bash
# Qwen3-8B
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/math.yaml \
  model_name_or_path=Qwen/Qwen3-8B-Base \
  per_device_train_batch_size=4 \
  gradient_accumulation_steps=16 \
  output_dir=saves/p3d/qwen3_8b_math

# Llama-3-8B
FORCE_TORCHRUN=1 NPROC_PER_NODE=8 llamafactory-cli train \
  ../P3D/configs/math.yaml \
  model_name_or_path=meta-llama/Meta-Llama-3-8B \
  template=llama3 \
  per_device_train_batch_size=4 \
  gradient_accumulation_steps=16 \
  output_dir=saves/p3d/llama3_8b_math
```

For medical data, use `medical.yaml` and accumulation `4` with micro-batch size
`4`, preserving an effective batch size of 128 on 8 GPUs.

</details>

## License

This code is distributed under the Apache License, Version 2.0 (full text below).
The loss retains its original anonymous copyright notice. Release edits add
argument checks and a differentiable zero for fully masked batches; the target
construction, numerical guards, and reduction are unchanged.

The integration targets the public upstream
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) project at the commit
specified above. Its copyright is held by HuggingFace Inc. and the LlamaFactory
team (2025). The DeepSpeed configuration is adapted from its
`examples/deepspeed/ds_z3_config.json`. These names identify upstream software,
not the authors of P³D. Retain upstream notices in the LLaMA-Factory checkout.

<details>
<summary>Apache License 2.0 — full text</summary>

```text
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

</details>

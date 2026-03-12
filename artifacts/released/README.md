---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
pipeline_tag: text-generation
tags: [lora, tool-use, grpo, reinforcement-learning, scenetrace]
language: [en]
---
# SceneTrace tool router

Actual rank-8 LoRA weights trained on SmolLM2-135M-Instruct for five-way constrained
video-workspace tool routing. This is a small language-model adapter, not a trained VLM
or general-purpose autonomous video agent. CLIP retrieval is an independent frozen model.

[Live workspace](https://scenetrace-inioluwa.vercel.app) ·
[Source and inference implementation](https://github.com/Eunini/scenetrace) ·
[Training data](https://huggingface.co/datasets/Ineeoluwa/scenetrace-tool-traces)

## Actual experiment

Executed: 2026-10-28T01:16:24.058767+00:00. Base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
Three epochs of completion-only SFT, then 60 finite-action GRPO steps with group size six,
sampling temperature 3,
two clipped updates per rollout, exact routing reward, and an SFT-reference KL penalty.
Model selection used validation questions; the held-out test was evaluated afterward.
The released stage is **SFT**. Both comparison checkpoints
are included under `comparisons/sft` and `comparisons/grpo`. RL outcomes are reported even
if the gate selects SFT. Nothing was trained in the retrospective Git timestamp period.

| Stage | Correct held-out routes | Accuracy | Mean CPU routing latency |
|---|---:|---:|---:|
| BASE | 8/40 | 20.0% | 1.531s |
| SFT | 40/40 | 100.0% | 1.659s |
| GRPO | 40/40 | 100.0% | 1.610s |

All questions are synthetic. The 480/40/40 split is disjoint by question-template family;
this small benchmark does not establish real-user reliability or general reasoning gains.
See `evaluation.json`, `training-history.json`, source objectives, and per-example predictions
in the linked GitHub repository. Estimated cost uses an explicit assumed CPU-hour price,
not billing records. There are no claimed customer counts.

## Inference

Install the source project and use `scenetrace.policy.Planner` with this downloaded adapter
directory. Its pinned base model scores five fixed completions (`search`, `library`, `count`,
`export`, `inspect`); their length-normalized log-probabilities define the policy. Do not
expect unrestricted text generation to reproduce the constrained-routing evaluation.
The app validates each tool call and enforces export approval independently of the model.

## Limitations and licensing

English synthetic requests, limited context and five available routes. Confidence values are
uncalibrated categorical scores. No safety, medical, legal, surveillance, or broad visual
understanding claim is made. Base model and this adapter: Apache 2.0. Application code: MIT.
Model and evaluation checksums are in `release.json`.

# Training and evaluation protocol

This experiment post-trains the open-weight **SmolLM2-135M-Instruct** causal language model
with a rank-8 LoRA adapter on query/key-value projection subsets (q and v projections).
The task is **five-way constrained tool routing**, not general visual reasoning. The frozen
CLIP visual encoder is independent and receives no training updates.

## Data

`training/data.py` makes 560 synthetic, explicitly labelled routing requests: 480 train,
40 validation and 40 test. Each intent has 16 template families; 12 go to train, two to
validation and two to test. Variants of a template never cross splits. Some non-search
variants repeat surface text, which is disclosed rather than counted as independent
real-user examples. There are no customer questions, curated human trajectories or
claimed production users in this dataset.

The five plans are search, library inventory, library aggregate, export preparation and
object inspection. Each labelled trace names the bounded tools the route would invoke.
An export route requests an approval; it cannot authorize a render.

## SFT

Completion-only causal cross-entropy teaches the model to emit a route name after a bounded
request/context prompt. The vocabulary projection is computed only at supervised completion
positions, reducing CPU memory. A parity test checks that these scores match full causal LM
scores and preserve gradients. The base model stays frozen; only actual LoRA parameters
are updated. Training is seeded and run on CPU, not simulated.

## Finite-action GRPO

The language model assigns a length-normalized completion log-probability to each of the
five candidates; their softmax defines a categorical policy. For each rollout, six actions
are sampled from that current policy. Reward is +1 for the labelled correct route and -1
otherwise, minus 0.01 per expected tool call. Rewards are standardized within each group.
Two clipped policy updates use epsilon 0.2 and exact categorical KL to the frozen SFT
reference, weighted by beta 0.02. One hundred rollout steps are executed.

This is a finite-action group-relative policy optimization experiment. It does not use
token-level open-ended generation, train a VLM, optimize answers using a judge, or claim
equivalence to large-scale GRPO systems. The deterministic routing labels verify the reward;
visual evidence quality is tested separately. The full supervised and policy objectives,
hyperparameters, losses and measured outcomes are available in source and JSON artifacts.

## Evaluation gate

Base, SFT and GRPO are evaluated with exact route matching. The candidate is selected on
validation accuracy, before the held-out test is evaluated. A release requires validation
accuracy at least 0.75 and held-out accuracy no worse than the base model. If GRPO regresses
on validation, SFT is served and the regression remains in the published report. There is
no claim that RL necessarily improves this small, already supervised task.

The report includes per-intent accuracy, every prediction, average routing latency and
estimated dollars per correct route. Cost uses observed CPU seconds at an explicit assumed
$0.04/hour and excludes storage, idle hosting, indexing, networking and capital costs. It is
not billing data or end-to-end dollars per correct video answer. Forty held-out questions
are a small synthetic benchmark; no broad reliability claim follows from a high score.

API and browser integration tests exercise real video decoding, retrieval, timestamps and
approved MP4 exports. Production smoke checks run through the deployed UI with actual
models. These establish working product flows, not external adoption or independently
calibrated human evaluation. Real customer trials and consented failure data are the next
evidence needed before expanding the model claim.

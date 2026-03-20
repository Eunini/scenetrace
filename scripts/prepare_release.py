"""Package measured artifacts and their provenance; never manufacture missing metrics."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "artifacts/evaluation.json").read_text())
    assert report["release_gate"]["passed"]
    directory = ROOT / "artifacts/released"
    for stage in ["sft", "grpo"]:
        shutil.copytree(
            ROOT / f"artifacts/checkpoints/{stage}",
            directory / f"comparisons/{stage}",
            dirs_exist_ok=True,
        )
        metric = report["metrics"][f"{stage}_test"]
        (directory / f"comparisons/{stage}/README.md").write_text(
            f"# SceneTrace {stage.upper()} comparison checkpoint\n\n"
            f"Actual rank-8 LoRA adapter for five-way tool routing. Held-out synthetic test: "
            f"{metric['correct']}/{metric['n']} correct. "
            f"The validation-selected served stage is {report['selected_stage'].upper()}.\n\n"
            "See the parent model card and evaluation.json for training, provenance and limitations. "
            "Base and adapter license: Apache 2.0.\n"
        )
    for folder in [directory, directory / "comparisons/sft", directory / "comparisons/grpo"]:
        config_path = folder / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["revision"] = report["base_revision"]
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    release = json.loads((directory / "release.json").read_text())
    release["adapter_sha256"] = hashlib.sha256(
        (directory / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    release["data_sha256"] = {
        s: hashlib.sha256((ROOT / f"training/{s}.jsonl").read_bytes()).hexdigest()
        for s in ["train", "validation", "test"]
    }
    release["comparison_sha256"] = {
        stage: hashlib.sha256(
            (directory / f"comparisons/{stage}/adapter_model.safetensors").read_bytes()
        ).hexdigest()
        for stage in ["sft", "grpo"]
    }
    (directory / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    rows = []
    for stage in ["base", "sft", "grpo"]:
        metric = report["metrics"][f"{stage}_test"]
        rows.append(
            f"| {stage.upper()} | {metric['correct']}/{metric['n']} | {metric['accuracy']:.1%} | {metric['mean_latency_seconds']:.3f}s |"
        )
    table = "\n".join(rows)
    card = f"""---
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

Executed: {report["executed_at"]}. Base revision: `{report["base_revision"]}`.
Three epochs of completion-only SFT, then 60 finite-action GRPO steps with group size six,
sampling temperature 3,
two clipped updates per rollout, exact routing reward, and an SFT-reference KL penalty.
Model selection used validation questions; the held-out test was evaluated afterward.
The released stage is **{report["selected_stage"].upper()}**. Both comparison checkpoints
are included under `comparisons/sft` and `comparisons/grpo`. RL outcomes are reported even
if the gate selects SFT. Nothing was trained in the retrospective Git timestamp period.

| Stage | Correct held-out routes | Accuracy | Mean CPU routing latency |
|---|---:|---:|---:|
{table}

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
"""
    (directory / "README.md").write_text(card)
    for name in ["evaluation.json", "training-history.json", "sft-provenance.json"]:
        shutil.copyfile(ROOT / "artifacts" / name, directory / name)
    shutil.copyfile(ROOT / "licenses/Apache-2.0.txt", directory / "LICENSE")
    output = ROOT / "training/release"
    output.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
    for split in ["train", "validation", "test"]:
        shutil.copyfile(ROOT / f"training/{split}.jsonl", output / f"{split}.jsonl")
    (output / "README.md").write_text("""---
license: mit
task_categories: [text-classification, text-generation]
language: [en]
tags: [tool-use, synthetic, scenetrace]
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---
# SceneTrace tool-routing traces

560 synthetic, labelled English routing requests for an open-weight video-workspace planner.
480 train, 40 validation, 40 test; question-template families never cross splits. Some
non-search variants repeat surface text, so these are not 560 independent human examples.

Fields: id, template_group, query, selected (whether a frame is selected), previous (empty in
this dataset), plan (one of five route names), source, tools (expected bounded tool sequence).
The dataset contains route labels and intended tool traces, not recorded customer sessions,
human-written demonstrations, model-generated free-form trajectories or private uploads.

[Generator, training code and evaluation](https://github.com/Eunini/scenetrace).
Actual data creation and training occurred in October 2026; the repository's January–March
Git dates were assigned retrospectively. There are no asserted real-user adoption counts.
""")
    print(json.dumps({"stage": report["selected_stage"], "packaged": True, "test_results": table}))


if __name__ == "__main__":
    main()

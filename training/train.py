"""Real LoRA SFT and finite-action GRPO. No generated answers are graded by a model."""

import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from peft import LoraConfig, TaskType, get_peft_model
from scenetrace.policy import (
    BASE_MODEL,
    BASE_REVISION,
    PLANS,
    encode_options,
    option_scores,
    prompt_for,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
torch.set_num_threads(2)
torch.manual_seed(37)
random.seed(37)


def load_data(split):
    return [
        json.loads(line) for line in (ROOT / f"training/{split}.jsonl").read_text().splitlines()
    ]


def encoded(tokenizer, rows):
    return encode_options(
        tokenizer, [prompt_for(r["query"], r["selected"], r["previous"]) for r in rows]
    )


def evaluate(model, tokenizer, rows):
    predictions, latencies = [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            start = time.perf_counter()
            scores = option_scores(model, encoded(tokenizer, [row]))[0]
            predicted = PLANS[int(scores.argmax())]
            latencies.append(time.perf_counter() - start)
            predictions.append(
                {
                    "id": row["id"],
                    "expected": row["plan"],
                    "predicted": predicted,
                    "correct": predicted == row["plan"],
                }
            )
    correct = sum(r["correct"] for r in predictions)
    seconds = sum(latencies)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_latency_seconds": seconds / len(rows),
        "estimated_dollars_per_correct": seconds / 3600 * 0.04 / max(correct, 1),
        "cost_assumption": "CPU compute at $0.04/hour; excludes storage and idle time, not a bill",
        "per_plan": {
            p: sum(r["correct"] for r in predictions if r["expected"] == p)
            / sum(r["expected"] == p for r in predictions)
            for p in PLANS
        },
        "predictions": predictions,
    }


def completion_batch(tokenizer, rows):
    records = []
    for row in rows:
        prompt = tokenizer.encode(
            prompt_for(row["query"], row["selected"]), add_special_tokens=False
        )
        answer = tokenizer.encode(" " + row["plan"], add_special_tokens=False)
        records.append((prompt + answer, [-100] * len(prompt) + answer))
    width = max(len(r[0]) for r in records)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, (tokens, targets) in enumerate(records):
        ids[i, : len(tokens)] = torch.tensor(tokens)
        labels[i, : len(tokens)] = torch.tensor(targets)
        attention[i, : len(tokens)] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def reward(row, action):
    """Executable exact routing reward with a cost penalty; exporting always requests approval."""
    correct = PLANS[action] == row["plan"]
    calls = 2 if PLANS[action] == "search" else 1
    return (1.0 if correct else -1.0) - 0.01 * calls


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    train, validation, test = [load_data(s) for s in ["train", "validation", "test"]]
    groups = [{r["template_group"] for r in rows} for rows in [train, validation, test]]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        use_safetensors=True,
    )
    report = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "seed": 37,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "task": "Five-way constrained video-tool routing, not free-form video reasoning",
        "split_sizes": dict(
            zip(["train", "validation", "test"], map(len, [train, validation, test]))
        ),
        "split_policy": "Disjoint question template groups; all questions are synthetic",
        "training": {},
        "metrics": {},
    }
    print("Evaluating base on validation", flush=True)
    report["metrics"]["base_validation"] = evaluate(model, tokenizer, validation)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-4)
    history = []
    resume_sft = "--resume-sft" in sys.argv
    if resume_sft:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(
            model, load_file(OUT / "checkpoints/sft/adapter_model.safetensors")
        )
        provenance = json.loads((OUT / "sft-provenance.json").read_text())
        history.extend(provenance["completed_epoch_losses"])
        report["resumed_sft"] = provenance
    for epoch in range(3):
        model.train()
        random.shuffle(train)
        if resume_sft:
            continue
        losses = []
        for start in range(0, len(train), 8):
            batch = completion_batch(tokenizer, train[start : start + 8])
            # Project only completion positions into the 49k-token vocabulary. This is
            # exactly completion-masked causal cross-entropy with far less CPU memory.
            base = model.get_base_model()
            hidden = base.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).last_hidden_state[:, :-1]
            targets = batch["labels"][:, 1:]
            selected = targets != -100
            logits = base.get_output_embeddings()(hidden[selected])
            loss = torch.nn.functional.cross_entropy(logits.float(), targets[selected])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        point = {"stage": "sft", "epoch": epoch + 1, "loss": sum(losses) / len(losses)}
        history.append(point)
        print(json.dumps(point), flush=True)
    if not resume_sft:
        model.save_pretrained(OUT / "checkpoints/sft", safe_serialization=True)
    report["training"]["sft"] = {"epochs": 3, "batch_size": 8, "learning_rate": 8e-4}
    report["metrics"]["sft_validation"] = evaluate(model, tokenizer, validation)
    print("SFT validation", report["metrics"]["sft_validation"]["accuracy"], flush=True)

    # Finite-action GRPO: sample six tool plans from the current LM policy, normalize
    # rewards within the group, then apply two clipped updates with exact categorical KL.
    model.eval()
    temperature = 3.0
    initial_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rl_rows = random.sample(train, 80)
    reference = []
    with torch.no_grad():
        for row in rl_rows:
            reference.append(
                torch.log_softmax(
                    option_scores(model, encoded(tokenizer, [row]))[0] / temperature, -1
                )
            )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-5)
    informative_groups = 0
    for step in range(60):
        i = step % len(rl_rows)
        row, ref = rl_rows[i], reference[i]
        inputs = encoded(tokenizer, [row])
        model.eval()
        with torch.no_grad():
            old = torch.log_softmax(option_scores(model, inputs)[0] / temperature, -1)
            actions = torch.multinomial(old.exp(), 6, replacement=True)
            rewards = torch.tensor([reward(row, int(a)) for a in actions])
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-4)
            informative_groups += int(float(rewards.std(unbiased=False)) > 1e-4)
        for _ in range(2):
            current = torch.log_softmax(option_scores(model, inputs)[0] / temperature, -1)
            ratios = (current[actions] - old[actions]).exp()
            objective = torch.minimum(
                ratios * advantages, ratios.clamp(0.8, 1.2) * advantages
            ).mean()
            kl = (current.exp() * (current - ref)).sum()
            loss = -objective + 0.02 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            point = {
                "stage": "grpo",
                "step": step,
                "reward": float(rewards.mean()),
                "kl": float(kl.detach()),
            }
            history.append(point)
            print(json.dumps(point), flush=True)
    model.save_pretrained(OUT / "checkpoints/grpo", safe_serialization=True)
    report["training"]["grpo"] = {
        "steps": 60,
        "sampling_temperature": temperature,
        "informative_reward_groups": informative_groups,
        "adapter_l2_change": sum(
            float(((p.detach() - initial_lora[n]) ** 2).sum())
            for n, p in model.named_parameters()
            if n in initial_lora
        )
        ** 0.5,
        "group_size": 6,
        "updates_per_rollout": 2,
        "clip_epsilon": 0.2,
        "beta": 0.02,
        "learning_rate": 8e-5,
        "action_space": list(PLANS),
        "reward": "+1 correct plan, -1 wrong plan; -0.01 per tool call",
    }
    report["metrics"]["grpo_validation"] = evaluate(model, tokenizer, validation)
    selected = (
        "grpo"
        if report["metrics"]["grpo_validation"]["accuracy"]
        >= report["metrics"]["sft_validation"]["accuracy"]
        else "sft"
    )
    report["selected_stage"] = selected
    # Freeze model selection before evaluating the held-out test. No test-based tuning.
    from peft import PeftModel

    base = model.unload()
    report["metrics"]["base_test"] = evaluate(base, tokenizer, test)
    for stage in ["sft", "grpo"]:
        candidate = PeftModel.from_pretrained(base, OUT / f"checkpoints/{stage}")
        report["metrics"][f"{stage}_test"] = evaluate(candidate, tokenizer, test)
        base = candidate.unload()
    report["release_gate"] = {
        "passed": report["metrics"][f"{selected}_validation"]["accuracy"] >= 0.75
        and report["metrics"][f"{selected}_test"]["accuracy"]
        >= report["metrics"]["base_test"]["accuracy"],
        "rule": "Validation accuracy >= 0.75; frozen candidate must not underperform base on held-out test",
    }
    report["elapsed_seconds"] = time.time() - started
    (OUT / "training-history.json").write_text(json.dumps(history, indent=2) + "\n")
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["release_gate"]["passed"]:
        destination = OUT / "released"
        shutil.copytree(OUT / f"checkpoints/{selected}", destination, dirs_exist_ok=True)
        tokenizer.save_pretrained(destination)
        release = {
            "stage": selected,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "actual_training_time": report["executed_at"],
            "evaluation_sha256": hashlib.sha256((OUT / "evaluation.json").read_bytes()).hexdigest(),
        }
        (destination / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ["selected_stage", "release_gate", "elapsed_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()

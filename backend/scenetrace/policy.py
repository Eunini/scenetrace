"""A small causal LM selects among five typed plans, never arbitrary Python or SQL."""

from pathlib import Path
import threading

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PLANS = ("search", "library", "count", "export", "inspect")
ROOT = Path(__file__).resolve().parents[2]


def prompt_for(query: str, selected: bool = False, previous: str = "") -> str:
    return (
        "Choose the video workspace tool plan.\n"
        "search: find visual scenes; library: list videos; count: total duration or quantity; "
        "export: prepare a clip download; inspect: identify objects in the selected frame.\n"
        f"Selected clip: {'yes' if selected else 'no'}. "
        f"Previous search: {previous[:100] or 'none'}.\n"
        f"Request: {query[:400]}\nPlan:"
    )


def encode_options(tokenizer, prompts):
    import torch

    rows, lengths = [], []
    for prompt in prompts:
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        for plan in PLANS:
            suffix = tokenizer.encode(" " + plan, add_special_tokens=False)
            rows.append(prefix + suffix)
            lengths.append((len(prefix), len(prefix) + len(suffix)))
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    targets = torch.zeros((len(rows), width - 1), dtype=torch.float32)
    for i, (row, (start, end)) in enumerate(zip(rows, lengths, strict=True)):
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
        targets[i, start - 1:end - 1] = 1
    return ids, mask, targets


def option_scores(model, encoded):
    """Mean completion log-probabilities; categorical policy is softmax(scores)."""
    import torch

    ids, attention, targets = encoded
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
    token_logps = torch.log_softmax(logits.float(), -1).gather(
        -1, ids[:, 1:, None]
    ).squeeze(-1)
    scores = (token_logps * targets).sum(-1) / targets.sum(-1).clamp(min=1)
    return scores.reshape(-1, len(PLANS))


class Planner:
    def __init__(self, adapter: Path | None = None):
        self.adapter = adapter or ROOT / "artifacts/released"
        self.lock = threading.Lock()
        self.model = self.tokenizer = None
        self.stage = "unavailable"

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, torch_dtype=torch.float32,
            attn_implementation="eager", use_safetensors=True,
        )
        if not (self.adapter / "adapter_config.json").is_file():
            raise RuntimeError("The evaluated planner adapter is not installed.")
        self.model = PeftModel.from_pretrained(model, self.adapter).eval()
        import json
        manifest = self.adapter / "release.json"
        self.stage = json.loads(manifest.read_text())["stage"] if manifest.exists() else "adapter"

    def choose(self, query: str, selected=False, previous=""):
        import torch

        with self.lock, torch.inference_mode():
            self.load()
            scores = option_scores(
                self.model, encode_options(self.tokenizer, [prompt_for(query, selected, previous)])
            )[0]
            probabilities = torch.softmax(scores, -1).tolist()
        best = max(range(len(PLANS)), key=probabilities.__getitem__)
        return {
            "plan": PLANS[best], "stage": self.stage,
            "scores": dict(zip(PLANS, [round(p, 4) for p in probabilities], strict=True)),
            "model": BASE_MODEL,
        }

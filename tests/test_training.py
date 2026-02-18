import json
from pathlib import Path

import pytest
import torch
from scenetrace.policy import PLANS, option_scores
from scenetrace.tools import LibraryArgs, ToolBudget, ToolError
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def test_training_splits_do_not_share_question_template_groups():
    sets = []
    for split in ("train", "validation", "test"):
        rows = [json.loads(r) for r in (ROOT / f"training/{split}.jsonl").read_text().splitlines()]
        assert set(r["plan"] for r in rows) == set(PLANS)
        assert all(r["source"] == "synthetic-routing-template" for r in rows)
        sets.append({r["template_group"] for r in rows})
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_memory_efficient_completion_scores_equal_full_causal_lm_scores():
    torch.set_num_threads(1)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).eval()
    ids = torch.randint(0, 32, (5, 8))
    attention = torch.ones_like(ids)
    targets = torch.zeros((5, 7))
    targets[:, 5:] = 1
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
    logps = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    expected = ((logps * targets).sum(-1) / targets.sum(-1)).reshape(-1, 5)
    actual = option_scores(model, (ids, attention, targets))
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    assert model.get_output_embeddings().weight.grad is not None


def test_tool_budget_stops_execution_and_errors_are_visible():
    budget = ToolBudget(1)
    assert budget.call("query_library", lambda: 3) == 3
    with pytest.raises(ToolError, match="budget"):
        budget.call("query_library", lambda: 4)
    assert budget.used == 1
    failing = ToolBudget()

    def fail():
        raise ValueError("No frame selected")

    with pytest.raises(ToolError, match="No frame"):
        failing.call("retrieve_frame", fail)
    assert failing.trace[0]["status"] == "error"


def test_library_accepts_only_typed_parameterized_modes():
    with pytest.raises(ValueError):
        LibraryArgs(mode="SELECT * FROM videos")

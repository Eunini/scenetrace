"""Validate the released planner, split provenance and actual artifact checksums."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    report_path = ROOT / "artifacts/evaluation.json"
    report = json.loads(report_path.read_text())
    release = json.loads((ROOT / "artifacts/released/release.json").read_text())
    if not report["release_gate"]["passed"]:
        raise ValueError("The trained planner has not passed its evaluation gate.")
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != release["evaluation_sha256"]:
        raise ValueError("The evaluation report changed after model selection.")
    if release["stage"] != report["selected_stage"]:
        raise ValueError("The released adapter does not match the evaluated candidate.")
    assert (ROOT / "artifacts/released/adapter_model.safetensors").stat().st_size > 1000
    if release.get("adapter_sha256"):
        assert (
            hashlib.sha256(
                (ROOT / "artifacts/released/adapter_model.safetensors").read_bytes()
            ).hexdigest()
            == release["adapter_sha256"]
        )
    for split, checksum in release.get("data_sha256", {}).items():
        assert (
            hashlib.sha256((ROOT / f"training/{split}.jsonl").read_bytes()).hexdigest() == checksum
        )
    for name, value in report["metrics"].items():
        assert value["n"] == len(value["predictions"])
        assert value["correct"] == sum(p["correct"] for p in value["predictions"])
        assert value["accuracy"] == value["correct"] / value["n"]
    print(
        json.dumps(
            {"release_gate": "passed", "stage": release["stage"], "base": release["base_model"]}
        )
    )


if __name__ == "__main__":
    verify()

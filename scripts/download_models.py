"""Download pinned publisher files for a CPU deployment. Never read or print credentials."""

from huggingface_hub import snapshot_download
from scenetrace.policy import BASE_MODEL, BASE_REVISION
from scenetrace.vision import CLIP_MODEL, CLIP_REVISION, DETR_MODEL, DETR_REVISION

if __name__ == "__main__":
    for model, revision, weights in [
        (BASE_MODEL, BASE_REVISION, "model.safetensors"),
        (CLIP_MODEL, CLIP_REVISION, "pytorch_model.bin"),
        (DETR_MODEL, DETR_REVISION, "pytorch_model.bin"),
    ]:
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=[weights, "*.json", "*.txt", "*.model", "*.jinja", "LICENSE*"],
        )
        print(f"Cached {model} at {revision}", flush=True)

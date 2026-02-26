"""Publish this evaluated release using the pre-existing private Hugging Face login."""

import json
import secrets
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from release_gate import verify

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "Ineeoluwa"


def main():
    verify()
    api = HfApi()
    if api.whoami().get("name") != ACCOUNT:
        raise SystemExit("The intended account is Ineeoluwa. No publication performed.")
    destinations = [
        ("model", "scenetrace-router-135m", ROOT / "artifacts/released"),
        ("dataset", "scenetrace-tool-traces", ROOT / "training/release"),
        ("space", "scenetrace", ROOT),
    ]
    for kind, name, directory in destinations:
        repo = f"{ACCOUNT}/{name}"
        new = False
        try:
            info = api.repo_info(repo, repo_type=kind)
            marker = hf_hub_download(repo, ".scenetrace-release.json", repo_type=kind)
            if json.loads(Path(marker).read_text()).get("project") != "Eunini/scenetrace":
                raise SystemExit(f"Existing repository has a different owner marker: {repo}")
        except RepositoryNotFoundError:
            api.create_repo(
                repo,
                repo_type=kind,
                private=False,
                **({"space_sdk": "docker"} if kind == "space" else {}),
            )
            info = api.repo_info(repo, repo_type=kind)
            new = True
        (directory / ".scenetrace-release.json").write_text(
            json.dumps({"project": "Eunini/scenetrace"}) + "\n"
        )
        if kind == "space":
            if new:
                api.add_space_secret(repo, "SCENETRACE_SESSION_SECRET", secrets.token_urlsafe(48))
            api.add_space_variable(
                repo, "SCENETRACE_PUBLIC_ORIGIN", "https://scenetrace-inioluwa.vercel.app"
            )
            api.upload_folder(
                repo_id=repo,
                repo_type=kind,
                folder_path=directory,
                allow_patterns=[
                    "README.md",
                    "LICENSE",
                    "NOTICE.md",
                    "Dockerfile",
                    ".dockerignore",
                    ".scenetrace-release.json",
                    "pyproject.toml",
                    "uv.lock",
                    "backend/**",
                    "web/**",
                    "assets/**",
                    "artifacts/released/**",
                    "artifacts/evaluation.json",
                    "artifacts/training-history.json",
                    "scripts/download_models.py",
                    "docs/**",
                ],
                ignore_patterns=[
                    "**/node_modules/**",
                    "**/__pycache__/**",
                    "web/dist/**",
                    "web/test-results/**",
                    "web/playwright-report/**",
                    "*.tsbuildinfo",
                ],
                parent_commit=info.sha,
                commit_message="Release the evaluated SceneTrace video workspace",
            )
        else:
            api.upload_folder(
                repo_id=repo,
                repo_type=kind,
                folder_path=directory,
                parent_commit=info.sha,
                commit_message="Publish reproducible SceneTrace training artifacts",
            )
        print(
            json.dumps(
                {"repo": repo, "type": kind, "commit": api.repo_info(repo, repo_type=kind).sha}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

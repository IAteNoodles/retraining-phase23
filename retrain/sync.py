"""HuggingFace sync: upload checkpoints, download for resume.

Resume supports `hf://<repo>/<path>` pointers, e.g.
  --resume_checkpoint hf://username/repo/phase_2/checkpoint-500
which is snapshot-downloaded to a local cache dir first.
"""

from __future__ import annotations

import os


def require_hub():
    try:
        from huggingface_hub import HfApi, login, snapshot_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required for sync (pip install huggingface_hub)") from e
    return HfApi, login, snapshot_download


def upload_dir(local_dir: str, repo_id: str, path_in_repo: str, token: str = "") -> None:
    _, login, _ = require_hub()
    from huggingface_hub import upload_folder
    if token:
        login(token=token)
    upload_folder(folder_path=local_dir, repo_id=repo_id, path_in_repo=path_in_repo,
                  repo_type="model", commit_message=f"sync {path_in_repo}")
    print(f"uploaded {local_dir} -> {repo_id}/{path_in_repo}")


def resolve_checkpoint(spec: str, token: str = "", cache_dir: str = "./hf_resume_cache") -> str:
    """Local path passthrough, or hf://repo/path download."""
    if not spec:
        return ""
    if not spec.startswith("hf://"):
        if not os.path.isdir(spec):
            raise FileNotFoundError(f"resume checkpoint not found: {spec!r}")
        return spec
    _, login, snapshot_download = require_hub()
    if token:
        login(token=token)
    rest = spec[len("hf://"):]
    repo_id, _, path = rest.partition("/")
    if not repo_id or not path:
        raise ValueError(f"bad hf checkpoint spec (want hf://repo/path): {spec!r}")
    local = snapshot_download(repo_id=repo_id, repo_type="model",
                              allow_patterns=[path.rstrip("/") + "/*"],
                              local_dir=cache_dir, local_dir_use_symlinks=False,
                              token=token or None)
    out = os.path.join(local, path)
    if not os.path.isdir(out):
        raise FileNotFoundError(f"downloaded checkpoint missing: {out!r}")
    print(f"resolved {spec} -> {out}")
    return out

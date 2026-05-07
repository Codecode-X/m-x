#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="${VENV_DIR:-/home/xiaojunhao/.venvs/monet_isolated}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found at: $PYTHON_BIN" >&2
  exit 1
fi

HF_DATASET_ID="${HF_DATASET_ID:-Kwai-Keye/Thyme-RL}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/Thyme-RL}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

check_dataset_complete() {
  "$PYTHON_BIN" - <<PY
from pathlib import Path
from huggingface_hub import HfApi
import sys

repo_id = "${HF_DATASET_ID}"
data_root = Path(r"${DATA_ROOT}")

if not data_root.exists():
    print(f"Local dir does not exist: {data_root}")
    sys.exit(1)

try:
    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)
except Exception as e:
    print(f"Failed to fetch dataset metadata: {e}")
    sys.exit(1)

# 过滤目录占位，仅校验真实文件
remote_files = [s for s in info.siblings if s.rfilename and not s.rfilename.endswith("/")]
if not remote_files:
    print("No remote files discovered; cannot verify dataset.")
    sys.exit(1)

missing = []
size_mismatch = []
checked = 0

for s in remote_files:
    rel = s.rfilename
    lp = data_root / rel
    if not lp.exists() or lp.stat().st_size <= 0:
        missing.append(rel)
        continue

    # 只在拿得到远端大小时比对，避免 metadata 缺失误判
    if s.size is not None and lp.stat().st_size != s.size:
        size_mismatch.append((rel, lp.stat().st_size, s.size))
    checked += 1

if missing:
    print("Missing or empty files:")
    for x in missing[:50]:
        print("  -", x)
    if len(missing) > 50:
        print(f"  ... and {len(missing)-50} more")
    sys.exit(1)

if size_mismatch:
    print("Size mismatch files:")
    for rel, ls, rs in size_mismatch[:50]:
        print(f"  - {rel}: local={ls}, remote={rs}")
    if len(size_mismatch) > 50:
        print(f"  ... and {len(size_mismatch)-50} more")
    sys.exit(1)

print(f"Dataset verification passed: {data_root} (checked {checked} files)")
sys.exit(0)
PY
}

mkdir -p "$(dirname "$DATA_ROOT")"

if [[ "$FORCE_DOWNLOAD" != "1" ]]; then
  if check_dataset_complete; then
    echo "Dataset already complete at $DATA_ROOT"
    exit 0
  else
    echo "Dataset is incomplete. Resume downloading..." >&2
  fi
fi

"$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_DATASET_ID}",
    repo_type="dataset",
    local_dir=r"${DATA_ROOT}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY

if check_dataset_complete; then
  echo "Downloaded and verified: ${HF_DATASET_ID} -> ${DATA_ROOT}"
else
  echo "Download finished but dataset is still incomplete or corrupted." >&2
  exit 1
fi

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

HF_DATASET_ID="${HF_DATASET_ID:-NOVAglow646/Monet-SFT-125K}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/Monet-SFT-125K}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

REQUIRED_SUBSETS=(
  "Visual_CoT"
  "CogCoM"
  "ReFocus"
  "Zebra_CoT_count"
  "Zebra_CoT_visual_search"
  "Zebra_CoT_geometry"
)

check_dataset_complete() {
  "$PYTHON_BIN" - <<PY
from pathlib import Path
from huggingface_hub import HfApi
import sys

repo_id = "${HF_DATASET_ID}"
data_root = Path(r"${DATA_ROOT}")
# 显式列表，避免 shell 展开造成 Python 语法错误。
subsets = [
    "Visual_CoT",
    "CogCoM",
    "ReFocus",
    "Zebra_CoT_count",
    "Zebra_CoT_visual_search",
    "Zebra_CoT_geometry",
]

required_rel = []
for s in subsets:
    required_rel.append(f"{s}/train.json")
    required_rel.append(f"{s}/images.zip")

missing = []
size_mismatch = []

# 先做本地存在性检查
for rel in required_rel:
    p = data_root / rel
    if not p.exists() or p.stat().st_size <= 0:
        missing.append(rel)

if missing:
    print("Missing or empty files:")
    for x in missing:
        print("  -", x)
    sys.exit(1)

# 再做远端大小校验（重点是 images.zip）
try:
    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)
    remote_size = {}
    for s in info.siblings:
        if s.rfilename in required_rel:
            remote_size[s.rfilename] = s.size

    for rel in required_rel:
        # 如果远端拿不到 size，就跳过该文件的 size 校验（不影响存在性）
        rs = remote_size.get(rel)
        if rs is None:
            continue
        lp = data_root / rel
        ls = lp.stat().st_size

        # 对 images.zip 强制比对大小；train.json 也一并比对更稳
        if ls != rs:
            size_mismatch.append((rel, ls, rs))

except Exception as e:
    # 若 metadata 拉取失败，保守返回不完整，避免误判
    print(f"Failed to fetch remote metadata: {e}")
    sys.exit(1)

if size_mismatch:
    print("Size mismatch files:")
    for rel, ls, rs in size_mismatch:
        print(f"  - {rel}: local={ls}, remote={rs}")
    sys.exit(1)

print(f"Dataset verification passed: {data_root}")
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
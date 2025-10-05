#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Ensure we're running from project root for predictable relative paths.
cd "$ROOT"

function banner() {
  printf '\n==== %s ====' "$1"
  printf '\n'
}

banner "pytest"
pytest "$@"

banner "CLI smoke checks"
python tools/calibration/calibrate_intrinsics.py --help >/dev/null
python tools/calibration/analyze_calibration.py --help >/dev/null
python tools/evaluation/evaluate_board.py --help >/dev/null
python tools/augmentation/data_aug.py --help >/dev/null

if [[ -n "${DATASET_DIR:-}" ]]; then
  banner "minimal calibration example"
  python local_sandbox/minimal_calibration_example.py \
    --dataset "$DATASET_DIR" \
    --output "${ROOT}/outputs/local_sandbox_calibration" \
    --limit 8
else
  banner "minimal calibration example"
  echo "DATASET_DIR 未设置，跳过基于真实数据的最小示例。"
  echo "如需运行，请执行: DATASET_DIR=/path/to/calib_images local_sandbox/run_all.sh"
fi

banner "done"

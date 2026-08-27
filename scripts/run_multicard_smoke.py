# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the multicard smoke test and print a human-readable summary.

For real tensor-parallel multi-card runs, launch with torchrun (one process
per card).  For single-card, plain python works.

Requirements:
  - transformers==5.15.x  (uv pip install "transformers==5.15.0")
  - HF_DEACTIVATE_ASYNC_LOAD=1  (disables async weight loading, required for TP with transformers 5.15)

AIU_IDS is a comma-separated list of PCI addresses for the cards to use.
Set it to match the cards available in your environment.

2-card example::

    export AIU_IDS="<card0-pci>,<card1-pci>"
    export HF_DEACTIVATE_ASYNC_LOAD=1
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py

4-card example::

    export AIU_IDS="<card0-pci>,<card1-pci>,<card2-pci>,<card3-pci>"
    export HF_DEACTIVATE_ASYNC_LOAD=1
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=4 --master-port=29500 \\
        scripts/run_multicard_smoke.py

Single-card::

    python scripts/run_multicard_smoke.py

The --model argument accepts any HuggingFace repo ID or local path
(default: ibm-granite/granite-3.3-8b-instruct).

The script exits with code 0 on PASS and code 1 on FAIL or ERROR.

Note: a ``corrupted double-linked list`` / SIGABRT crash may appear after the
RESULTS SUMMARY prints.  This is a known shutdown bug in the Spyre runtime
(libsenlib-dd2.so destructors) and does not affect correctness — if the
summary shows ``Status: PASS`` the test passed.
"""

import argparse
import os
import sys

# Ensure the project root (parent of scripts/) is on sys.path so that
# tests.spyre.test_multicard_spyre can be imported when running directly
# from anywhere inside the repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.spyre.test_multicard_spyre import run_multicard_smoke_test  # noqa: E402

DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_MAX_NEW_TOKENS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path or local dir (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Token generation budget (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    return parser


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}{unit}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    aiu_ids = os.environ.get("AIU_IDS")

    print()
    print("=" * 70)
    print("  run_multicard_smoke")
    print("=" * 70)
    print(f"  AIU_IDS          : {aiu_ids!r}")
    if aiu_ids:
        cards = [c.strip() for c in aiu_ids.split(",") if c.strip()]
        print(f"  Cards detected   : {len(cards)}")
        for i, addr in enumerate(cards):
            print(f"    [{i}] {addr}")
    else:
        print("  Cards detected   : AIU_IDS not set — single-card / default behaviour")
    print(f"  Model            : {args.model}")
    print(f"  max_new_tokens   : {args.max_new_tokens}")
    print("=" * 70)

    result = run_multicard_smoke_test(args.model, args.max_new_tokens)

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  AIU_IDS    : {result['aiu_ids_env']!r}")
    print(f"  Model      : {result['model']}")
    print(f"  Status     : {result['status']}")
    print(f"  Load       : {_fmt(result['load_s'], 's')}")
    print(f"  Generate   : {_fmt(result['gen_s'], 's')}")
    print(f"  TTFT       : {_fmt(result['ttft_ms'], ' ms')}")
    print(f"  Decode avg : {_fmt(result['decode_ms'], ' ms')}")
    print(f"  Steady ITL : {_fmt(result['steady_itl_ms'], ' ms')}  (outliers excluded)")
    print(f"  Output     : {result['output']!r}")
    if result["error"]:
        print()
        print("  ERROR DETAIL:")
        for line in result["error"].splitlines():
            print(f"    {line}")
    print("=" * 70)
    print()

    os._exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

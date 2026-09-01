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

SPYRE_DEVICES tells the Flex runtime which card indices to use per rank.
Index-to-physical-card mapping is handled internally by Flex and is not
independently verifiable from this script.

2-card example::

    export SPYRE_DEVICES=2,3
    export HF_DEACTIVATE_ASYNC_LOAD=1
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

4-card example::

    export SPYRE_DEVICES=0,1,2,3
    export HF_DEACTIVATE_ASYNC_LOAD=1
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=4 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

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

import torch

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
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Torch dtype to pass to AutoSpyreModelForCausalLM.from_pretrained "
            "(e.g. float16, bfloat16, float32).  Omit to let the model decide."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="Number of identical prompts to batch together (default: 1).",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        metavar="L",
        help=(
            "Pad the tokenized prompt to exactly L tokens before generation. "
            "Omit to use the natural token length."
        ),
    )
    return parser


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}{unit}"


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dtype is not None:
        if args.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown --dtype {args.dtype!r}. "
                f"Choose from: {', '.join(_DTYPE_MAP)}"
            )
    dtype = _DTYPE_MAP.get(args.dtype) if args.dtype is not None else None

    aiu_ids = os.environ.get("AIU_IDS")
    spyre_devices = os.environ.get("SPYRE_DEVICES")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Trim AIU_IDS to WORLD_SIZE entries so a 4-address env var doesn't get
    # printed when only 1 or 2 ranks are actually launched.
    if aiu_ids is not None:
        _ids = [c.strip() for c in aiu_ids.split(",") if c.strip()]
        if len(_ids) > world_size:
            aiu_ids = ",".join(_ids[:world_size])

    print()
    print("=" * 70)
    print("  run_multicard_smoke")
    print("=" * 70)
    print(f"  SPYRE_DEVICES    : {spyre_devices!r}")
    print(f"  AIU_IDS          : {aiu_ids!r}  (fallback; SPYRE_DEVICES is primary)")
    print(f"  Model            : {args.model}")
    print(f"  max_new_tokens   : {args.max_new_tokens}")
    print(f"  batch            : {args.batch}")
    print(f"  prompt_len       : {args.prompt_len if args.prompt_len is not None else '(natural)'}")
    print(f"  dtype            : {args.dtype or '(not set — model default)'}")
    print("=" * 70)

    result = run_multicard_smoke_test(
        args.model,
        args.max_new_tokens,
        dtype=dtype,
        batch_size=args.batch,
        prompt_len=args.prompt_len,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  SPYRE_DEVICES : {result['spyre_devices_env']!r}")
    if result['resolved_cards']:
        print(f"  PCI hint      : {', '.join(result['resolved_cards'])}  (AIU_WORLD_RANK_* guess, unconfirmed)")
    print(f"  AIU_IDS       : {result['aiu_ids_env']!r}")
    print(f"  Model         : {result['model']}")
    print(f"  Batch      : {result['batch_size']}")
    print(f"  Prompt len : {result['prompt_len'] if result['prompt_len'] is not None else '(natural)'}")
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

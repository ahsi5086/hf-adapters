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

"""
Multicard tensor-parallel smoke test for hf-adapters on Spyre AIU cards.

Verifies that a model sharded across N cards via real tensor parallelism:
  - Loads correctly on every rank
  - Produces non-empty output containing the expected text
  - Reports TTFT and steady-state ITL (compile-spike outliers excluded)

This test must be launched with torchrun — one process per card.  Regular
pytest cannot drive multi-process TP because each rank needs its own OS
process with its own LOCAL_RANK and device assignment.

Requirements
------------
- transformers==5.15.x  (5.16 changed the tp_plan API; see pyproject.toml)
- HF_DEACTIVATE_ASYNC_LOAD=1  (disables async weight loading, required for TP with transformers 5.15)

How to run
----------
First make sure transformers is at the right version::

    uv pip install "transformers==5.15.0"

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

pytest (single-card only — multi-card requires torchrun, see above)::

    pytest -s -vvv tests/spyre/test_multicard_spyre.py
"""

import contextlib
import io
import os
import re
import statistics
import time
import traceback
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct"
_PROMPT = "The capital of France is"
_EXPECTED_SUBSTRING = "Paris"
_DEFAULT_MAX_NEW_TOKENS = 8

# Compile-spike outlier filter: tokens whose latency exceeds this multiple of
# the median are excluded from the steady-state ITL figure.  This catches the
# second-token decode-graph compile spike without hardcoding a token index.
_OUTLIER_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _parse_timing(captured: str) -> tuple[float | None, float | None, list[float]]:
    """Return (ttft_ms, avg_decode_ms, per_token_list) from captured stdout."""
    ttft = None
    decode = None
    per_token: list[float] = []

    m = re.search(r"First-token latency:\s*([\d.]+)\s*ms", captured)
    if m:
        ttft = float(m.group(1))

    m = re.search(r"Avg next-token latency:\s*([\d.]+)\s*ms", captured)
    if m:
        decode = float(m.group(1))

    m = re.search(r"Per-token:\s*([\d.,\s]+)\s*ms", captured)
    if m:
        per_token = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]

    return ttft, decode, per_token


def _steady_state_itl(per_token: list[float]) -> float | None:
    """Mean of post-TTFT tokens after excluding compile-spike outliers.

    ``per_token[0]`` is TTFT and is always excluded.  Any remaining token
    whose latency exceeds ``_OUTLIER_THRESHOLD * median`` is also excluded.
    Returns None when fewer than 2 decode tokens are available.
    """
    if len(per_token) < 2:
        return None
    decode = per_token[1:]  # drop TTFT
    if len(decode) == 1:
        return decode[0]
    median = statistics.median(decode)
    if median <= 0:
        return statistics.mean(decode)
    steady = [v for v in decode if v <= _OUTLIER_THRESHOLD * median]
    return statistics.mean(steady) if steady else None


# ---------------------------------------------------------------------------
# Core smoke-test function (used by both the CLI script and pytest)
# ---------------------------------------------------------------------------


def run_multicard_smoke_test(
    model_path: str, max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
) -> dict[str, Any]:
    """Load model under the current AIU_IDS env, generate tokens, return diagnostics.

    Load and generation are wrapped in separate try/except blocks so a failure
    clearly identifies whether it occurred at load time or generate time, with
    the full traceback captured rather than swallowed.

    Returns a dict with keys:
        model           - model path used
        aiu_ids_env     - value of AIU_IDS at call time (None if unset)
        local_rank      - LOCAL_RANK of this process
        world_size      - WORLD_SIZE of this run
        status          - "PASS" | "FAIL" | "ERROR"
        load_s          - seconds spent in from_pretrained (None on load error)
        gen_s           - seconds spent in generate() (None on gen error)
        ttft_ms         - first-token latency in ms (or None)
        decode_ms       - avg next-token latency in ms (or None)
        steady_itl_ms   - steady-state ITL with outliers removed (or None)
        output          - generated text (empty string on any error)
        error           - "PHASE FAILED\\n<traceback>" string, or None on PASS
    """
    # torch MUST be imported before torch_spyre.  torch_spyre registers itself
    # as a PyTorch backend at import time; importing it before torch triggers a
    # circular import error.
    import torch  # noqa: F401  — must precede any torch_spyre import

    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM

    aiu_ids_env = os.environ.get("AIU_IDS")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"\n{'=' * 70}")
    print(f"  Multicard smoke test  [rank {local_rank}/{world_size}]")
    print(f"  Model         : {model_path}")
    print(f"  AIU_IDS       : {aiu_ids_env!r}")
    print(f"  LOCAL_RANK    : {local_rank}")
    print(f"  WORLD_SIZE    : {world_size}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"{'=' * 70}")

    result: dict[str, Any] = {
        "model": model_path,
        "aiu_ids_env": aiu_ids_env,
        "local_rank": local_rank,
        "world_size": world_size,
        "status": "ERROR",
        "load_s": None,
        "gen_s": None,
        "ttft_ms": None,
        "decode_ms": None,
        "steady_itl_ms": None,
        "output": "",
        "error": None,
    }

    # ── Phase 1: model load ────────────────────────────────────────────────
    model = None
    tokenizer = None
    load_t0 = time.time()
    try:
        tp = "auto" if world_size > 1 else None
        model = AutoSpyreModelForCausalLM.from_pretrained(model_path, tp_plan=tp)
        result["load_s"] = time.time() - load_t0
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  Load time  : {result['load_s']:.1f}s  [OK]")
    except Exception:
        result["load_s"] = time.time() - load_t0
        result["error"] = "LOAD FAILED\n" + traceback.format_exc()
        print(f"  Load FAILED (after {result['load_s']:.1f}s):\n{result['error']}")
        return result

    # ── Phase 2: generation ────────────────────────────────────────────────
    print(f"  Prompt     : {_PROMPT!r}")
    gen_t0 = time.time()
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            outputs = model.generate(
                tokenizer,
                [_PROMPT],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                timing=True,
            )
        result["gen_s"] = time.time() - gen_t0
        captured = buf.getvalue()
        print(captured, end="")  # echo timing lines to the caller's stdout

        ttft, decode, per_token = _parse_timing(captured)
        result["ttft_ms"] = ttft
        result["decode_ms"] = decode
        result["steady_itl_ms"] = _steady_state_itl(per_token)

        output_text = outputs[0] if outputs else ""
        result["output"] = output_text
        print(f"  Output     : {output_text!r}")
        print(f"  Gen time   : {result['gen_s']:.1f}s  [OK]")
        if result["steady_itl_ms"] is not None:
            print(f"  Steady ITL : {result['steady_itl_ms']:.1f} ms  (outliers excluded)")
        result["status"] = "PASS" if output_text.strip() else "FAIL"
    except Exception:
        result["gen_s"] = time.time() - gen_t0
        result["error"] = "GENERATE FAILED\n" + traceback.format_exc()
        print(
            f"  Generate FAILED (after {result['gen_s']:.1f}s):\n{result['error']}"
        )

    return result


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------
# NOTE: This test runs in a single process (no torchrun).  It exercises the
# single-card code path (tp_plan=None, WORLD_SIZE=1).  Multi-card TP requires
# torchrun; use scripts/run_multicard_smoke.py for that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_path", [_DEFAULT_MODEL])
def test_multicard_smoke_single_card(model_path: str) -> None:
    """Single-card smoke test: load, generate, verify output contains expected text."""
    result = run_multicard_smoke_test(model_path)
    assert result["status"] == "PASS", (
        f"Smoke test failed with status {result['status']}.\n"
        f"Error: {result['error']}"
    )
    assert _EXPECTED_SUBSTRING.lower() in result["output"].lower(), (
        f"Expected {_EXPECTED_SUBSTRING!r} in output, got {result['output']!r}"
    )


# ---------------------------------------------------------------------------
# Standalone entry point (called by scripts/run_multicard_smoke.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _model = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_MODEL
    _max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_MAX_NEW_TOKENS
    _result = run_multicard_smoke_test(_model, _max_new_tokens)
    print(f"\nFinal status: {_result['status']}")
    if _result["error"]:
        print(_result["error"])
        raise SystemExit(1)

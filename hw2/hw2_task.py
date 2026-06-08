import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    generated_tokens = []

    # Prefill: process full prompt, populate KV cache
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens.append(next_token_id)

    # Decode: one token at a time, reusing cached K/V
    for _ in range(n_steps - 1):
        outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token_id)

    return [t.item() for t in generated_tokens]


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.float16)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. float32 -> float16: 4.78x speedup (9.41s -> 1.97s on T4).
#    Halves memory traffic and unlocks fp16 tensor core gemm paths
#    (volta_sgemm -> turing_fp16_s1688gemm). CUDA time per 12 steps
#    dropped from 883ms to 231ms.
#
# 2. inference_mode + remove per-step .item() sync: 4.78x -> 4.91x.
#    Eliminates CPU-GPU synchronization each step and skips autograd
#    bookkeeping. CUDA time dropped from 231ms to 180ms (22% reduction).
#    Modest gain because matmul kernels dominate, not sync overhead.
#
# 3. KV cache: 36.53x total (14.49s -> 0.40s, 323 tok/s on T4).
#    Each decode step now processes 1 token instead of the full growing
#    sequence. Eliminates O(n^2) redundant attention recomputation.
#    CUDA time per 12 steps dropped from 1274ms to 45ms. Matmuls became
#    matrix-vector ops (gemvx) instead of full gemm.
#
# Biggest impact and why:
#
# KV cache for sure. The baseline re-ran attention over all prior tokens every
# step with O(n) complexity. Caching K/V turns it into O(1) per new token.

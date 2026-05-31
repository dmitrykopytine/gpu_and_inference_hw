import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
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
    # See the Writeup at the bottom of this file for the optimizations applied here.
    generated_tokens = []

    # Prefill: process the whole prompt once and prime the KV cache.
    outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    past_key_values = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    generated_tokens.append(next_token_id)

    # Decode: feed only the newest token; positions are inferred from the cache.
    for _ in range(n_steps - 1):
        outputs = model(
            input_ids=next_token_id.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)

    return torch.cat(generated_tokens).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    del model
    torch.cuda.empty_cache()
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
#   1. No per-step .item() sync: token IDs stay on the GPU and are copied back
#      to the host in a single .tolist() after the loop, removing sync overhead.
#      Impact: 1.02x (baseline-bound by GPU matmuls, so the sync removal alone
#      barely moved the clock — see step 2 for why).
#
#   2. KV cache (use_cache=True): reduces the per-step matmuls down to a single
#      vector, eliminating the redundant full-sequence recompute.
#      Impact: 1.02x -> 6.25x (~6.13x incremental). Biggest win — see below.
#
#   3. torch.inference_mode(): model.eval() only disables dropout etc.; autograd
#      still records a graph, causing CPU overhead.
#      Impact: 6.25x -> 8.78x (~1.40x incremental).
#
#   4. logits_to_keep=1: we only read logits[:, -1, :], so there's no need to
#      compute the lm_head over every prompt position.
#      Impact: no visible change.
#
# Also tried but reverted: reducing the model dtype — fp16, bf16, and TF32. In
# every case there was a minor but stably observed degradation (~10% slower),
# not a speedup. It would also be a precision tradeoff: it changes the numerics
# and can change the generated tokens.
#
# Biggest impact and why:
#
#   The KV cache (step 2) by far. The V0 baseline reprocesses the entire growing
#   sequence on every step, so the per-step matmul cost scales with the full
#   sequence length (~1024+ tokens) and the total work is quadratic in the
#   number of tokens instead of being linear.

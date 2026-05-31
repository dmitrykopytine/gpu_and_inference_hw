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
    # Three optimizations stacked here:
    #  1. KV cache (use_cache=True): the prompt is encoded once during prefill,
    #     then each decode step feeds only the *single* new token plus the cached
    #     keys/values. This turns each step's matmuls from O(seq_len) down to O(1),
    #     eliminating the redundant full-sequence recompute that dominated the GPU.
    #  2. No per-step .item() sync: token IDs stay on the GPU and are copied back to
    #     the host in a single .tolist() after the loop, preserving CPU/GPU overlap.
    #  3. torch.inference_mode(): model.eval() only disables dropout etc.; autograd
    #     still records a graph and bumps tensor version counters on every op. With
    #     hundreds of tiny ops per decode step that bookkeeping is pure CPU overhead.
    #     inference_mode disables it entirely — bit-identical output, lower CPU cost.
    #  4. logits_to_keep=1: we only ever read logits[:, -1, :], so there's no need to
    #     run the lm_head over every prompt position. During prefill that turns a
    #     [1024, 2048] x [2048, vocab] projection into [1, 2048] x [2048, vocab].
    #     Lossless — the discarded positions were never used.
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
#
# Biggest impact and why:
#

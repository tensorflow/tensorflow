"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "491b0483ac16baa85561a499c5ee754bf5e8a96e"
XPU_TRITON_SHA256 = "a4fc57e44982fbd2eab6019c63b63e24d807b2708bbc299e95aa252d62571c85"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

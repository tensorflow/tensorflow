"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "e6c9092c0febda4b41701460c3b652fd12e1cd01"
XPU_TRITON_SHA256 = "cdd72de76e9f1411f65cac412c002c9219766e189b95073f6889c52a6e9016a6"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

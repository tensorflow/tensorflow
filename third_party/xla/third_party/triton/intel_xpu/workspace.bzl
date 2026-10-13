"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "616ef4c791a0feb79243689b985ca2a2afd9fa6b"
XPU_TRITON_SHA256 = "e51b47dd8f6a59bd3574964337fc8b1deef04c34a2ff8fbe2a4ba2cba134f1cf"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

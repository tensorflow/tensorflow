"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "6046064c56a27193423475e0a6f412a08b13c6e6"
XPU_TRITON_SHA256 = "158213c620f555aaf295613f48240c63a6b40dbe7db3805f97f4c6f2361b7f7d"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "94739e788a55a765adefc6205a715d98a5833d47"
XPU_TRITON_SHA256 = "081b630569aa71b14842f9c5be3c641df7a33ad67f169ea339f6119d49dda0cb"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

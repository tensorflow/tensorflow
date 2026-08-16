"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "81a7d8ade32081289dcc423540a11a8bf325c515"
XPU_TRITON_SHA256 = "26bc614931a62e50d43622f4de93acc3c05fa2dba23cbad6a18edabff38e01bf"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

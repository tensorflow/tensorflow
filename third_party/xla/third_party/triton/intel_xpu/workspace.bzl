"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "409229b4bf21dcd130598e3cebdc6e4a0d3965ba"
XPU_TRITON_SHA256 = "66f7caeedf8a2d65e381309f4c5f792557dd660a55030b18f2ad69e3de5389ec"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

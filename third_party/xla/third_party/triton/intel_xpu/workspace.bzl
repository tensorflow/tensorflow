"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "2d4184c6f96005f78ae3fa6869a4fd2adfb70183"
XPU_TRITON_SHA256 = "4d1d59d070cfbe2921cad04238fc0e46d7016c6eca1aa53524dd71547e4b6107"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

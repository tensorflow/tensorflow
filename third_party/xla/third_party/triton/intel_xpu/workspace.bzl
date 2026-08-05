"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "7239f0f2331ec9d7f594aa750c618cc15580ffc0"
XPU_TRITON_SHA256 = "797c2b32d7dd7236d382d3652448f8b9127576883e907de717f0a36701238916"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

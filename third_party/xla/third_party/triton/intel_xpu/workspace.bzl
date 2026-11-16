"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "94ebf57b17997e3f863ae49d8e902d5362f0e27b"
XPU_TRITON_SHA256 = "caac797ac92046e04d9c22914e17f684a8e15aebdd6b31057ab4615c2fee7a10"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

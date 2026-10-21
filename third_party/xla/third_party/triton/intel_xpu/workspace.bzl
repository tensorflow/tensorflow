"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "8fe763b706c924a25f0f8f8340de8dc21e3ce289"
XPU_TRITON_SHA256 = "f2a88a924237744c63a2f9db9ecdd5401d81be8c8d54018ae065b60dee45e86d"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

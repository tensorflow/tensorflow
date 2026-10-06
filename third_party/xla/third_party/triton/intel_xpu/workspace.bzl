"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "9b507bc72f26026254e572c8d9e59c22497ae522"
XPU_TRITON_SHA256 = "98f63c0a7fbbf5c6a13a76fe976e3d2f352d731f56bf75c8043fa77e302a9772"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

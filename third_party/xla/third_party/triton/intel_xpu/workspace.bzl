"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "1a7c1031ac5e9bc39fc5e56c47b6df23af99c4f8"
XPU_TRITON_SHA256 = "975670b40d279f9e433cbf073aff12d353024b3e1ff8b2cf77e618ceb0aa1d96"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

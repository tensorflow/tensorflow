"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "499fef53942f7b879700e3e8e4518cb2d77aaadc"
XPU_TRITON_SHA256 = "a105c5a8fd180dee680305f4057af2a65101b3b58fd6ae38e5ed529be2764d3f"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

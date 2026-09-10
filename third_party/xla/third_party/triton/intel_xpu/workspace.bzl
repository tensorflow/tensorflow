"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "1056cab6ec1fce6bb07cf75bcfb8cad12ffcf13d"
XPU_TRITON_SHA256 = "6dfca0270f87ec2fc25b1151484ebc8abb4b682ee0cf053b7a658fb5e3bfc379"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

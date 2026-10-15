"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "e45abef5f100e02e2f9e98f5f5a2fd8dfe3c6c34"
XPU_TRITON_SHA256 = "a0d191feab8e91bc710f96fb0aefde8a22b0c99daac4f2c4153adc8f6c98b26c"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

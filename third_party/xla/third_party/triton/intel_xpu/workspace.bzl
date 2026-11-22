"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "dda5e10578e805570e30d2e9fbd0ff19463255ec"
XPU_TRITON_SHA256 = "5216a950ddcf9c4a9fc17c90f8e04943ee52f274a8dbe223614f8ea54a3c1b81"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

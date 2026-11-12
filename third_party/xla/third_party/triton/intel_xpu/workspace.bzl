"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "c7d67a3f61945cf67f0a5a466a32b78992bb395a"
XPU_TRITON_SHA256 = "f30eb403a561908ce84e45634fc0e7d64c0d47fe6540aa36945044765ec758bc"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

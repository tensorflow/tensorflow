"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "07d2e3d3e250d0f6cd5b374514c28c3ae9e3afd4"
XPU_TRITON_SHA256 = "3b2dbfeb4c7610e127c64d70f8034c71454c98070cb446bae6d4c7a54a05c558"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

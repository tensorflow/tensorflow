"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "33f9cb578f45b8fb719c728896b2a75f29e66616"
XPU_TRITON_SHA256 = "856f7c73cd5dd43a9157709f4c17ca70fbb1e2504803711e66d62a9e0dadadd1"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"

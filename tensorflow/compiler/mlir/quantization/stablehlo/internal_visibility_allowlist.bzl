"""Internal visibility rules."""

def internal_visibility_allowlist():
    """Returns a list of g3 packages that can depend on internal targets."""
    return [
        "//learning/brain/experimental/mlir/quantization/...",
        "//learning/brain/mlir/quantization/tensorflow/...",
        "//learning/brain/mobile/programmability/...",
        "//learning/brain/experimental/tfq/...",
    ]

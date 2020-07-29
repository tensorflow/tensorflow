"""Defines build macros for tensorflow kernels."""

def if_mlir_generated_gpu_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels:mlir_generated_gpu_kernels_enabled": if_true,
        "//conditions:default": if_false,
    })

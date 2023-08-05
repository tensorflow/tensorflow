"""Helpers for conditional XLA:GPU compilation."""

def if_gpu2(then, otherwise = []):
    return select({
        "//tensorflow/compiler/xla/mlir/backends/gpu2:enabled": then,
        "//conditions:default": otherwise,
    })

def if_not_gpu2(then, otherwise = []):
    return select({
        "//tensorflow/compiler/xla/mlir/backends/gpu2:enabled": otherwise,
        "//conditions:default": then,
    })

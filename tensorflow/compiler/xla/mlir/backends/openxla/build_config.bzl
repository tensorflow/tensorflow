"""Helpers for conditional OpenXLA compilation."""

def if_openxla(then, otherwise = []):
    return select({
        "//tensorflow/compiler/xla/mlir/backends/openxla:enabled": then,
        "//conditions:default": otherwise,
    })

def if_not_openxla(then, otherwise = []):
    return select({
        "//tensorflow/compiler/xla/mlir/backends/openxla:enabled": otherwise,
        "//conditions:default": then,
    })

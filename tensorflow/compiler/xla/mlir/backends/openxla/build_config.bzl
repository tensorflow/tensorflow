"""Helpers for conditional OpenXLA compilation."""

def if_openxla(then, otherwise = []):
    return select({
        "//tensorflow/compiler/xla/mlir/backends/openxla:with_openxla_compiler": then,
        "//conditions:default": otherwise,
    })

def if_not_openxla(then, otherwise = []):
    return select({
        "//tensorflow/compiler/xla/mlir/backends/openxla:with_openxla_compiler": otherwise,
        "//conditions:default": then,
    })

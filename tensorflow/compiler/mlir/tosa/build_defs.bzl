# Conditional build rules for TOSA libraries

def cond_cc_library(name, srcs, hdrs, deps, alwayslink):
    native.cc_library(
        name = name,
        srcs = select({
            "//tensorflow/compiler/mlir/tosa:enable-build": srcs,
            "//conditions:default": [],
            }),
        hdrs = select({
            "//tensorflow/compiler/mlir/tosa:enable-build": hdrs,
            "//conditions:default": [],
            }),
        deps = select({
            "//tensorflow/compiler/mlir/tosa:enable-build": deps,
            "//conditions:default": [],
            }),
        alwayslink = alwayslink,
        )

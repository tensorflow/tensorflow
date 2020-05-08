package(licenses = ["notice"])

py_library(
    name = "pywrap_tfd",
    srcs = ["pywrap_tfd.py"],
    deps = [
        "//tensorflow/python/tf_program/mlir_wrapper",
    ],
)

py_library(
    name = "mlir_gen",
    srcs = ["mlir_gen.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_tfd",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/types",
        "@gast_archive//:gast",
    ],
)

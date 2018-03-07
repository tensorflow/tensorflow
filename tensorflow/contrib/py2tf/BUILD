licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "py2tf",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/py2tf/impl",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/utils",
        "@gast_archive//:gast",
        "@six_archive//:six",
    ],
)

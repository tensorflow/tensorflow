package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:dataset_ops",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/experimental/service",
    ],
)

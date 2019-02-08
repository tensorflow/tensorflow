package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/contrib/data/python/ops:iterator_ops",
        "//tensorflow/python:util",
    ],
)

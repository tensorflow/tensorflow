licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "autograph",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    # This module is kept for backward compatibility only. To depend on AutoGraph,
    # use //third_party/tensorflow/python/autograph instead.
    deps = [
        "//tensorflow/python/autograph",
    ],
)

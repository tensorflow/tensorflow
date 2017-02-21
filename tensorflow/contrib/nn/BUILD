# Description:
#   Contains deprecated ops to calculate cross entropy.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nn_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
        "python/ops/cross_entropy.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python:nn"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

# Description:
#   Contains the Slim library, including common neural networks and examples.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "queues",
    srcs = ["python/ops/queues.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:ops",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "slim",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":queues",
    ],
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
    visibility = ["//tensorflow:__subpackages__"],
)

# Implementation of a prototype TF distributed computation library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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
    name = "distribute",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/contrib/distribute/python:cross_tower_ops",
        "//tensorflow/contrib/distribute/python:mirrored_strategy",
        "//tensorflow/contrib/distribute/python:monitor",
        "//tensorflow/contrib/distribute/python:one_device_strategy",
        "//tensorflow/contrib/distribute/python:step_fn",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
    ],
)

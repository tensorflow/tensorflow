licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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
    name = "operators",
    srcs = [
        "__init__.py",
        "control_flow.py",
        "data_structures.py",
        "slices.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/contrib/autograph/utils",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

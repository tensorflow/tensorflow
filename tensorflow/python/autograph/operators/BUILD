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
        "exceptions.py",
        "logical.py",
        "py_builtins.py",
        "slices.py",
        "special_values.py",
        "symbols.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/autograph/utils",
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
    tags = [
        "no_gpu",  # b/127001953
    ],
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "exceptions_test",
    srcs = ["exceptions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "py_builtins_test",
    srcs = ["py_builtins_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
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

py_test(
    name = "special_values_test",
    srcs = ["special_values_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "symbols_test",
    srcs = ["symbols_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

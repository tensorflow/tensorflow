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
    name = "utils",
    srcs = [
        "__init__.py",
        "context_managers.py",
        "misc.py",
        "multiple_dispatch.py",
        "py_func.py",
        "tensor_list.py",
        "tensors.py",
        "testing.py",
        "type_check.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/data/ops:dataset_ops",
        "@six_archive//:six",
    ],
)

py_test(
    name = "context_managers_test",
    srcs = ["context_managers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "misc_test",
    srcs = ["misc_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "multiple_dispatch_test",
    srcs = ["multiple_dispatch_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "py_func_test",
    srcs = ["py_func_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "type_check_test",
    srcs = ["type_check_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "tensor_list_test",
    srcs = ["tensor_list_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
    ],
)

py_test(
    name = "tensors_test",
    srcs = ["tensors_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

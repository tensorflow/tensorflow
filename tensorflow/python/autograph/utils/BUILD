load("//tensorflow:tensorflow.bzl", "py_test")

package(
    licenses = ["notice"],
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
    visibility = ["//visibility:private"],
)

py_library(
    name = "utils",
    srcs = [
        "__init__.py",
        "ag_logging.py",
        "context_managers.py",
        "misc.py",
        "py_func.py",
        "tensor_list.py",
        "tensors.py",
        "testing.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/data/ops:dataset_ops",
        "@six_archive//:six",
    ],
)

py_test(
    name = "context_managers_test",
    srcs = ["context_managers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "misc_test",
    srcs = ["misc_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "py_func_test",
    srcs = ["py_func_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "tensor_list_test",
    srcs = ["tensor_list_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
    ],
)

py_test(
    name = "tensors_test",
    srcs = ["tensors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:client_testlib",
    ],
)

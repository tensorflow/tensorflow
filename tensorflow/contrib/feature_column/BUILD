package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

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
    name = "feature_column_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequence_feature_column",
    ],
)

py_library(
    name = "sequence_feature_column",
    srcs = ["python/feature_column/sequence_feature_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
    ],
)

py_test(
    name = "sequence_feature_column_test",
    srcs = ["python/feature_column/sequence_feature_column_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":sequence_feature_column",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column",
        "//third_party/py/numpy",
    ],
)

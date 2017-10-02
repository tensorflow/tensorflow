package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "dataset_ops",
    srcs = [
        "dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":transformation_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator",
        "//tensorflow/python/data/util:nest",
    ],
)

py_library(
    name = "readers",
    srcs = [
        "readers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
    ],
)

py_library(
    name = "transformation_ops",
    srcs = [
        "batching.py",
        "enumerate_ops.py",
        "error_ops.py",
        "grouping.py",
        "resampling.py",
        "sloppy_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
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

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":iterator_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:random_seed",
        "//tensorflow/python/data/util:sparse",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "readers",
    srcs = ["readers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/util:convert",
    ],
)

py_library(
    name = "iterator_ops",
    srcs = ["iterator_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
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

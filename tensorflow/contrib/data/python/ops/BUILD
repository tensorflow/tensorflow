package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

py_library(
    name = "dataset_ops",
    srcs = [
        "counter.py",
        "get_single_element.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":transformation_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

py_library(
    name = "iterator_ops",
    srcs = [
        "iterator_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
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
        ":shuffle_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
    ],
)

py_library(
    name = "shuffle_ops",
    srcs = [
        "shuffle_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":random_ops",
        ":transformation_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "transformation_ops",
    srcs = [
        "batching.py",
        "enumerate_ops.py",
        "error_ops.py",
        "grouping.py",
        "interleave_ops.py",
        "resampling.py",
        "scan_ops.py",
        "sliding.py",
        "stats_ops.py",
        "threadpool.py",
        "unique.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":contrib_op_loader",
        ":gen_dataset_ops",
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
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//third_party/py/numpy",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_dataset_ops",
    out = "gen_dataset_ops.py",
    deps = ["//tensorflow/contrib/data:dataset_ops_op_lib"],
)

tf_kernel_library(
    name = "dataset_ops_kernels",
    deps = [
        "//tensorflow/contrib/data/kernels:dataset_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "contrib_op_loader",
    srcs = ["contrib_op_loader.py"],
    dso = ["//tensorflow/contrib/data:_dataset_ops.so"],
    kernels = [
        ":dataset_ops_kernels",
        "//tensorflow/contrib/data:dataset_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_dataset_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "prefetching_ops",
    srcs = ["prefetching_ops.py"],
    deps = [
        ":contrib_op_loader",
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

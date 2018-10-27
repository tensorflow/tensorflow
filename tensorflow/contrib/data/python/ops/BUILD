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
    name = "counter",
    srcs = ["counter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:counter",
    ],
)

py_library(
    name = "get_single_element",
    srcs = ["get_single_element.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:get_single_element",
    ],
)

py_library(
    name = "iterator_ops",
    srcs = [
        "iterator_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:iterator_ops",
    ],
)

py_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:random_ops",
    ],
)

py_library(
    name = "readers",
    srcs = [
        "readers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":batching",
        ":interleave_ops",
        ":parsing_ops",
        ":shuffle_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:experimental_dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "shuffle_ops",
    srcs = [
        "shuffle_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:shuffle_ops",
    ],
)

py_library(
    name = "batching",
    srcs = ["batching.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":get_single_element",
        ":grouping",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:convert",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "enumerate_ops",
    srcs = ["enumerate_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:enumerate_ops",
    ],
)

py_library(
    name = "error_ops",
    srcs = ["error_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:error_ops",
    ],
)

py_library(
    name = "grouping",
    srcs = ["grouping.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:grouping",
    ],
)

py_library(
    name = "interleave_ops",
    srcs = ["interleave_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:interleave_ops",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["parsing_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:parsing_ops",
    ],
)

py_library(
    name = "resampling",
    srcs = ["resampling.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:resampling",
    ],
)

py_library(
    name = "scan_ops",
    srcs = ["scan_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:scan_ops",
    ],
)

py_library(
    name = "sliding",
    srcs = ["sliding.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "threadpool",
    srcs = ["threadpool.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:threadpool",
    ],
)

py_library(
    name = "unique",
    srcs = [
        "unique.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:unique",
    ],
)

py_library(
    name = "writers",
    srcs = [
        "writers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/data/experimental/ops:writers",
    ],
)

py_library(
    name = "prefetching_ops",
    srcs = ["prefetching_ops.py"],
    deps = [
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
    ],
)

py_library(
    name = "dataset_ops",
    deps = [
        ":batching",
        ":counter",
        ":enumerate_ops",
        ":error_ops",
        ":get_single_element",
        ":grouping",
        ":interleave_ops",
        ":prefetching_ops",
        ":random_ops",
        ":readers",
        ":resampling",
        ":scan_ops",
        ":shuffle_ops",
        ":sliding",
        ":threadpool",
        ":unique",
        ":writers",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

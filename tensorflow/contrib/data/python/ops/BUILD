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
        ":scan_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "get_single_element",
    srcs = ["get_single_element.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
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
        ":batching",
        ":interleave_ops",
        ":shuffle_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
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
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "batching",
    srcs = ["batching.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "enumerate_ops",
    srcs = ["enumerate_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "error_ops",
    srcs = ["error_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":contrib_op_loader",
        ":gen_dataset_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "grouping",
    srcs = ["grouping.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "interleave_ops",
    srcs = ["interleave_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":contrib_op_loader",
        ":gen_dataset_ops",
        ":random_ops",
        "//tensorflow/contrib/stateless",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "resampling",
    srcs = ["resampling.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":batching",
        ":interleave_ops",
        ":scan_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "scan_ops",
    srcs = ["scan_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
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
    name = "stats_ops",
    srcs = ["stats_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "threadpool",
    srcs = ["threadpool.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":contrib_op_loader",
        ":gen_dataset_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "unique",
    srcs = [
        "unique.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":contrib_op_loader",
        ":gen_dataset_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
    ],
)

py_library(
    name = "writers",
    srcs = [
        "writers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
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
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/data/util:sparse",
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
        ":readers",
        ":resampling",
        ":scan_ops",
        ":shuffle_ops",
        ":sliding",
        ":stats_ops",
        ":threadpool",
        ":unique",
        ":writers",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

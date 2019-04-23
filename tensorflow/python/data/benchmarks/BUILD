package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "meta_benchmark",
    srcs = ["meta_benchmark.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "benchmark_base",
    srcs = ["benchmark_base.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "batch_benchmark",
    srcs = ["batch_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_base",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "filter_benchmark",
    srcs = ["filter_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "from_tensor_slices_benchmark",
    srcs = ["from_tensor_slices_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "list_files_benchmark",
    srcs = ["list_files_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "map_benchmark",
    srcs = ["map_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "range_benchmark",
    srcs = ["range_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

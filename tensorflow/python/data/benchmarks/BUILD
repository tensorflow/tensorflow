load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "benchmark_base",
    srcs = ["benchmark_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "meta_benchmark",
    srcs = ["meta_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:session",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "batch_benchmark",
    srcs = ["batch_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "filter_benchmark",
    srcs = ["filter_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "from_tensor_slices_benchmark",
    srcs = ["from_tensor_slices_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/experimental/ops:get_single_element",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "list_files_benchmark",
    srcs = ["list_files_benchmark.py"],
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

tf_py_test(
    name = "map_benchmark",
    srcs = ["map_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "prefetch_benchmark",
    srcs = ["prefetch_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "range_benchmark",
    srcs = ["range_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

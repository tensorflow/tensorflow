# Benchmarks for Keras preprocessing layers.
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

tf_py_test(
    name = "category_encoding_benchmark",
    srcs = ["category_encoding_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:category_encoding",
    ],
)

tf_py_test(
    name = "category_crossing_benchmark",
    srcs = ["category_crossing_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:category_crossing",
    ],
)

tf_py_test(
    name = "hashing_benchmark",
    srcs = ["hashing_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:hashing",
    ],
)

tf_py_test(
    name = "index_lookup_adapt_benchmark",
    srcs = ["index_lookup_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "index_lookup_forward_benchmark",
    srcs = ["index_lookup_forward_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "normalization_adapt_benchmark",
    srcs = ["normalization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:normalization",
    ],
)

tf_py_test(
    name = "discretization_adapt_benchmark",
    srcs = ["discretization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:discretization",
    ],
)

cuda_py_test(
    name = "image_preproc_benchmark",
    srcs = ["image_preproc_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:image_preprocessing",
    ],
)

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "assert_element_shape_test",
    srcs = ["assert_element_shape_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lmdb_dataset_op_test",
    size = "medium",
    srcs = ["lmdb_dataset_op_test.py"],
    data = ["//tensorflow/core:lmdb_testdata"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "reduce_dataset_test",
    size = "small",
    srcs = ["reduce_dataset_test.py"],
    deps = [
        "//tensorflow/contrib/data/python/ops:get_single_element",
        "//tensorflow/contrib/data/python/ops:grouping",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "slide_dataset_op_test",
    size = "small",
    srcs = ["slide_dataset_op_test.py"],
    deps = [
        "//tensorflow/contrib/data/python/ops:sliding",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

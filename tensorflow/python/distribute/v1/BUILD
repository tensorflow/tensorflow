load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cuda_py_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/182567920)
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:kernels",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:collective_util",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nccl_ops",
    ],
)

tf_py_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.py"],
    deps = [
        ":all_reduce",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//third_party/py/numpy",
    ],
)

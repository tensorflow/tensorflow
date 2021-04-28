# tf.data service tests.

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_py_test(
    name = "coordinated_read_ft_test",
    size = "medium",
    srcs = ["coordinated_read_ft_test.py"],
    shard_count = 16,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "coordinated_read_test",
    size = "medium",
    srcs = ["coordinated_read_test.py"],
    shard_count = 16,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "data_service_ops_test",
    size = "medium",
    srcs = ["data_service_ops_test.py"],
    shard_count = 32,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "distributed_epoch_test",
    size = "medium",
    srcs = ["distributed_epoch_test.py"],
    shard_count = 16,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "fault_tolerance_test",
    size = "small",
    srcs = ["fault_tolerance_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

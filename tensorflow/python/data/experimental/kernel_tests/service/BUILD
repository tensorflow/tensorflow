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

tf_py_test(
    name = "local_workers_test",
    size = "medium",
    srcs = ["local_workers_test.py"],
    shard_count = 24,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "multi_device_test",
    size = "small",
    srcs = ["multi_device_test.py"],
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
    name = "multi_process_cluster",
    srcs = ["multi_process_cluster.py"],
    srcs_version = "PY3",
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/distribute:multi_process_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_test(
    name = "multi_process_cluster_test",
    size = "medium",
    srcs = ["multi_process_cluster_test.py"],
    shard_count = 3,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
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

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "failure_handling_lib",
    srcs = [
        "failure_handling.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":gce_util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform",
        "//tensorflow/python/training:checkpoint_management",
    ],
)

py_library(
    name = "gce_util",
    srcs = [
        "gce_util.py",
    ],
    srcs_version = "PY3",
    deps = [
    ],
)

tf_py_test(
    name = "failure_handler_test",
    srcs = ["failure_handler_test.py"],
    shard_count = 4,
    deps = [
        ":failure_handling_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/module",
        "//tensorflow/python/platform",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:checkpoint_management",
        "//tensorflow/python/training/tracking:util",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "gce_failure_handler_test",
    srcs = ["gce_failure_handler_test.py"],
    shard_count = 4,
    tags = [
        "noasan",  # TODO(b/226154233): Flaky test
        "nomsan",  # TODO(b/226154233): Flaky test
    ],
    deps = [
        ":failure_handling_lib",
        ":gce_util",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
    ],
)

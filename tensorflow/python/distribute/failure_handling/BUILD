load("//tensorflow:tensorflow.default.bzl", "tf_custom_op_py_library", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        ":failure_handling_util",
        "//tensorflow/python:lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute/failure_handling:check_preemption_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "failure_handling_util",
    srcs = [
        "failure_handling_util.py",
    ],
    srcs_version = "PY3",
    deps = [
    ],
)

tf_py_test(
    name = "failure_handler_test",
    timeout = "long",
    srcs = ["failure_handler_test.py"],
    shard_count = 8,
    tags = [
        "no_windows",  # TODO(b/197981388)
    ],
    deps = [
        ":failure_handling_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
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
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "gce_failure_handler_test",
    srcs = ["gce_failure_handler_test.py"],
    shard_count = 8,
    tags = [
        "noasan",  # TODO(b/226154233): Flaky test
        "nomsan",  # TODO(b/226154233): Flaky test
    ],
    deps = [
        ":failure_handling_lib",
        ":failure_handling_util",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
    ],
)

tf_custom_op_py_library(
    name = "check_preemption_py",
    kernels = [
        "//tensorflow/core/distributed_runtime/preemption:check_preemption_op_kernel",
        "//tensorflow/core/distributed_runtime/preemption:check_preemption_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core/distributed_runtime/preemption:gen_check_preemption_op",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

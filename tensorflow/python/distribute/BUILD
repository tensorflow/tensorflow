package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "distribute",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":distribute_config",
        ":distribute_coordinator",
        ":distribute_coordinator_context",
    ],
)

py_library(
    name = "distribute_config",
    srcs = [
        "distribute_config.py",
    ],
    deps = [],
)

py_library(
    name = "distribute_coordinator",
    srcs = [
        "distribute_coordinator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_coordinator_context",
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "distribute_coordinator_test",
    size = "large",
    srcs = ["distribute_coordinator_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_pip",
        "notap",
    ],
    deps = [
        ":distribute_coordinator",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "distribute_coordinator_context",
    srcs = [
        "distribute_coordinator_context.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "multi_worker_util",
    srcs = [
        "multi_worker_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "multi_worker_util_test",
    srcs = ["multi_worker_util_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# Used only by estimator.
py_library(
    name = "estimator_training",
    srcs = [
        "estimator_training.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_coordinator",
        ":distribute_coordinator_context",
        "//tensorflow/python:training",
    ],
)

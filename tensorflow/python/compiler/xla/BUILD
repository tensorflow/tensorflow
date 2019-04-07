licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":xla",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "jit_test",
    size = "small",
    srcs = ["jit_test.py"],
    additional_deps = [
        ":compiler_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    xla_enabled = True,
)

py_library(
    name = "xla",
    srcs = ["xla.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/jit:xla_ops_py",
        "//tensorflow/compiler/jit/ops:xla_ops_grad",
        # Do not remove: required to run xla ops on Cloud.
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "xla_test",
    srcs = ["xla_test.py"],
    additional_deps = [
        ":xla",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:variable_scope",
    ],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
)

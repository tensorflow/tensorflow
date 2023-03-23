load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    srcs_version = "PY3",
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
    tags = [
        "no_windows",  # TODO(b/171385770)
    ],
    xla_enabled = True,
    deps = [
        ":compiler_py",
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
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "xla",
    srcs = ["xla.py"],
    srcs_version = "PY3",
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
        "//tensorflow/python/eager:def_function",
    ],
)

cuda_py_test(
    name = "xla_test",
    srcs = ["xla_test.py"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        ":xla",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/summary",
        "//tensorflow/python/tpu:tpu_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "jit_compile_test",
    srcs = ["jit_compile_test.py"],
    python_version = "PY3",
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
    ],
)

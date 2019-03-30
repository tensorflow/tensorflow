package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

licenses(["notice"])  # Apache 2.0

py_library(
    name = "parallel_for",
    srcs = [
        "__init__.py",
        "control_flow_ops.py",
        "gradients.py",
        "pfor.py",
        "test_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":gradients",
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "pfor_lib",
    srcs = ["pfor.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "control_flow_ops",
    srcs = ["control_flow_ops.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":pfor_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pfor_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "control_flow_ops_test",
    srcs = ["control_flow_ops_test.py"],
    additional_deps = [
        ":control_flow_ops",
        ":test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:util",
    ],
    tags = ["no_rocm"],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "array_test",
    srcs = ["array_test.py"],
    additional_deps = [
        ":control_flow_ops",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:backprop",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "math_test",
    srcs = ["math_test.py"],
    additional_deps = [
        ":control_flow_ops",
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:util",
    ],
    tags = ["optonly"],  # Too slow in non-opt mode
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "gradients",
    srcs = ["gradients.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "gradients_test",
    srcs = ["gradients_test.py"],
    additional_deps = [
        ":control_flow_ops",
        ":gradients",
        "//third_party/py/numpy",
        "//tensorflow/python:layers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/losses",
    ],
    tags = ["optonly"],  # Too slow in non-opt mode
    xla_enable_strict_auto_jit = True,
)

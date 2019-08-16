package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "mlir",
    srcs = ["mlir.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/mlir/python:mlir_extension",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "mlir_test",
    srcs = ["mlir_test.py"],
    python_version = "PY3",
    deps = [
        ":mlir",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/tpu:tpu_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "mlir",
    srcs = ["mlir.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "mlir_test",
    srcs = ["mlir_test.py"],
    python_version = "PY3",
    deps = [
        ":mlir",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

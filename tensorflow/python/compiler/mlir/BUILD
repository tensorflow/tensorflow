load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "mlir",
    srcs = ["mlir.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:pywrap_mlir",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "mlir_test",
    srcs = ["mlir_test.py"],
    data = [
        "multi_add.tflite",
    ],
    python_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":mlir",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/eager:def_function",
    ],
)

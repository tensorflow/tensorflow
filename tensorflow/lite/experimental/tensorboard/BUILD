# TFLite modules to support TensorBoard plugin.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "ops_util",
    srcs = ["ops_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python:wrap_toco",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "ops_util_test",
    srcs = ["ops_util_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":ops_util",
        "//tensorflow/python:client_testlib",
    ],
)

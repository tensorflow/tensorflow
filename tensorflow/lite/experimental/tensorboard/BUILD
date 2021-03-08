# TFLite modules to support TensorBoard plugin.

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "ops_util",
    srcs = ["ops_util.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python:wrap_toco",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "ops_util_test",
    srcs = ["ops_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops_util",
        "//tensorflow/python:client_testlib",
    ],
)

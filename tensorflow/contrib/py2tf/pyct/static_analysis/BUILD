licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "static_analysis",
    srcs = [
        "activity.py",
        "annos.py",
        "live_values.py",
        "type_info.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/py2tf/pyct",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "activity_test",
    srcs = ["activity_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "live_values_test",
    srcs = ["live_values_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "type_info_test",
    srcs = ["type_info_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":static_analysis",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

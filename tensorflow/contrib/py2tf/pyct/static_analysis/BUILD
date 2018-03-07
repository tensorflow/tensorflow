licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

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
        "//tensorflow/contrib/py2tf/utils",
        "//tensorflow/python:client_testlib",
    ],
)

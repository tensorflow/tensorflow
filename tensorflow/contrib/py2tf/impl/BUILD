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
    name = "impl",
    srcs = [
        "api.py",
        "config.py",
        "conversion.py",
        "naming.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/contrib/py2tf/converters",
        "//tensorflow/contrib/py2tf/pyct",
        "//tensorflow/contrib/py2tf/pyct/static_analysis",
        "@gast_archive//:gast",
        "@six_archive//:six",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":impl",
        "//tensorflow/contrib/py2tf/utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "conversion_test",
    srcs = ["conversion_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "naming_test",
    srcs = ["naming_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
    ],
)

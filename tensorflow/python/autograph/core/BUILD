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
    name = "core",
    srcs = [
        "config.py",
        "converter.py",
        "errors.py",
        "function_wrapping.py",
        "naming.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/autograph/utils",
    ],
)

py_library(
    name = "test_lib",
    srcs = [
        "converter_testing.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":core",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/autograph/utils",
        "@gast_archive//:gast",
        "@six_archive//:six",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "errors_test",
    srcs = ["errors_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
    ],
)

py_test(
    name = "function_wrapping_test",
    srcs = ["function_wrapping_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "naming_test",
    srcs = ["naming_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core",
        "//tensorflow/python:client_testlib",
    ],
)

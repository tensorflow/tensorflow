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
    name = "testing",
    srcs = [
        "codegen.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/contrib/autograph/utils",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "codegen_test",
    size = "large",
    srcs = ["codegen_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":testing",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

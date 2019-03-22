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
    name = "test_modules",
    srcs = [
        "future_import_module.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "codegen",
    srcs = [
        "codegen.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/utils",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "codegen_test",
    size = "large",
    srcs = ["codegen_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_windows",
        "nomsan",
        "notap",
    ],
    deps = [
        ":codegen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

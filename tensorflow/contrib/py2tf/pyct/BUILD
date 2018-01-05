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
    name = "pyct",
    srcs = [
        "__init__.py",
        "anno.py",
        "compiler.py",
        "parser.py",
        "pretty_printer.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "anno_test",
    srcs = ["anno_test.py"],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "pretty_printer_test",
    srcs = ["pretty_printer_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

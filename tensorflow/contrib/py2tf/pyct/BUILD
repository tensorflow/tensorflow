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
        "templates.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "@astor_archive//:astor",
        "@gast_archive//:gast",
        "@termcolor_archive//:termcolor",
    ],
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
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "pretty_printer_test",
    srcs = ["pretty_printer_test.py"],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "templates_test",
    srcs = ["templates_test.py"],
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

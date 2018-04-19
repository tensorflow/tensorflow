# Description:
#   Doc generator

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//tensorflow:__subpackages__"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "doc_generator_visitor",
    srcs = [
        "doc_generator_visitor.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "doc_generator_visitor_test",
    size = "small",
    srcs = [
        "doc_generator_visitor_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":doc_generator_visitor",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "parser",
    srcs = ["parser.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["@astor_archive//:astor"],
)

py_test(
    name = "parser_test",
    size = "small",
    srcs = ["parser_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":parser",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "pretty_docs",
    srcs = ["pretty_docs.py"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "generate_lib",
    srcs = ["generate_lib.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":doc_generator_visitor",
        ":parser",
        ":pretty_docs",
        ":py_guide_parser",
        "//tensorflow/contrib/ffmpeg:ffmpeg_ops_py",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
    ],
)

py_test(
    name = "generate_lib_test",
    size = "small",
    srcs = ["generate_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":generate_lib",
        ":parser",
        "//tensorflow/python:platform_test",
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":generate_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
    ],
)

py_test(
    name = "build_docs_test",
    size = "small",
    srcs = ["build_docs_test.py"],
    data = ["//tensorflow:docs_src"],
    srcs_version = "PY2AND3",
    tags = [
        # No reason to run sanitizers for this test.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":generate_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
    ],
)

py_binary(
    name = "generate_1_0",
    srcs = ["generate_1_0.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":generate_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
    ],
)

py_library(
    name = "py_guide_parser",
    srcs = ["py_guide_parser.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "py_guide_parser_test",
    size = "small",
    srcs = ["py_guide_parser_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":py_guide_parser",
        "//tensorflow/python:client_testlib",
    ],
)

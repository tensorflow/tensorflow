# Description:
#   Doc generator

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//tensorflow:__subpackages__"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

py_binary(
    name = "gen_cc_md",
    srcs = ["gen_cc_md.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

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
    srcs = [
        "parser.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "parser_test",
    size = "small",
    srcs = [
        "parser_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":parser",
        "//tensorflow/python:platform_test",
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":doc_generator_visitor",
        ":parser",
        ":py_guide_parser",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/ffmpeg:ffmpeg_ops_py",
        "//tensorflow/python/debug:debug_py",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
    ],
)

py_test(
    name = "generate_test",
    size = "small",
    srcs = [
        "generate_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":generate",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "py_guide_parser",
    srcs = [
        "py_guide_parser.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "py_guide_parser_test",
    size = "small",
    srcs = [
        "py_guide_parser_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":py_guide_parser",
        "//tensorflow/python:client_testlib",
    ],
)

filegroup(
    name = "doxy_config",
    srcs = ["tf-doxy_for_md-config"],
)

sh_binary(
    name = "gen_docs",
    srcs = ["gen_docs.sh"],
    data = [
        ":doxy_config",
        ":gen_cc_md",
        "//tensorflow/python:gen_docs_combined",
    ],
)

sh_test(
    name = "gen_docs_test",
    size = "small",
    srcs = [
        "gen_docs_test.sh",
    ],
    data = [
        ":gen_docs",
        "//tensorflow/core:all_files",
        "//tensorflow/python:all_files",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

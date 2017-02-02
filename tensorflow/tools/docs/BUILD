# Description:
#   Doc generator

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//tensorflow:__subpackages__"],
)

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
        "//tensorflow:tensorflow_py",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/docs:doc_generator_visitor",
        "//tensorflow/tools/docs:parser",
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

py_binary(
    name = "make_py_guides",
    srcs = ["make_py_guides.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/tools/docs:generate",
        "//tensorflow/tools/docs:parser",
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

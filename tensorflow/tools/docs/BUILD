# Description:
#   Doc generator

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "tf_doctest_lib",
    srcs = ["tf_doctest_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "tf_doctest",
    srcs = ["tf_doctest.py"],
    python_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_pip",
        "no_windows",  # numpy prints differently on windows.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":tf_doctest_lib",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "tf_doctest_test",
    srcs = ["tf_doctest_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_pip",
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":tf_doctest_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "doc_generator_visitor",
    srcs = [
        "doc_generator_visitor.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

py_test(
    name = "doc_generator_visitor_test",
    size = "small",
    srcs = [
        "doc_generator_visitor_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":doc_generator_visitor",
        ":generate_lib",
        "//tensorflow/python:platform_test",
        "@six_archive//:six",
    ],
)

py_library(
    name = "doc_controls",
    srcs = ["doc_controls.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "doc_controls_test",
    size = "small",
    srcs = ["doc_controls_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":doc_controls",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "parser",
    srcs = ["parser.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":doc_controls",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "@astor_archive//:astor",
        "@six_archive//:six",
    ],
)

py_test(
    name = "parser_test",
    size = "small",
    srcs = ["parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",  # UNIX filepath separators are hardcoded.
    ],
    deps = [
        ":parser",
        "//tensorflow/python:platform_test",
        "@six_archive//:six",
    ],
)

py_library(
    name = "pretty_docs",
    srcs = ["pretty_docs.py"],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

py_library(
    name = "generate_lib",
    srcs = ["generate_lib.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":doc_controls",
        ":doc_generator_visitor",
        ":parser",
        ":pretty_docs",
        ":py_guide_parser",
        "//tensorflow/python:util",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "@six_archive//:six",
    ],
)

py_test(
    name = "generate_lib_test",
    size = "small",
    srcs = ["generate_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",  # UNIX filepath separators are hardcoded.
    ],
    deps = [
        ":generate_lib",
        ":parser",
        "//tensorflow/python:platform_test",
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":generate_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:util",
        "//tensorflow/python/debug:debug_py",
    ],
)

py_test(
    name = "generate2_test",
    size = "medium",
    srcs = ["generate2_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        # No reason to run sanitizers or fastbuild for this test.
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
        "notap",
    ],
    deps = [
        ":generate2_lib",
    ],
)

py_binary(
    name = "generate2",
    srcs = ["generate2.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":generate2_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "generate2_lib",
    srcs = ["generate2.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "py_guide_parser",
    srcs = ["py_guide_parser.py"],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

py_test(
    name = "py_guide_parser_test",
    size = "small",
    srcs = ["py_guide_parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",  # Windows line endings break comparisons.
    ],
    deps = [
        ":py_guide_parser",
        "//tensorflow/python:client_testlib",
        "@six_archive//:six",
    ],
)

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

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
        "ast_util.py",
        "cfg.py",
        "error_utils.py",
        "errors.py",
        "gast_util.py",
        "inspect_utils.py",
        "loader.py",
        "origin_info.py",
        "parser.py",
        "pretty_printer.py",
        "qual_names.py",
        "templates.py",
        "transformer.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct/common_transformers",
        "@astor_archive//:astor",
        "@gast_archive//:gast",
        "@six_archive//:six",
        "@termcolor_archive//:termcolor",
    ],
)

py_test(
    name = "anno_test",
    srcs = ["anno_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "ast_util_test",
    srcs = ["ast_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "cfg_test",
    srcs = ["cfg_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "loader_test",
    srcs = ["loader_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "error_utils_test",
    srcs = ["error_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "inspect_utils_test",
    srcs = ["inspect_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct/testing",
        "@gast_archive//:gast",
    ],
)

sh_test(
    name = "inspect_utils_test_par",
    srcs = ["inspect_utils_test.sh"],
    tags = ["no_oss"],
)

py_test(
    name = "origin_info_test",
    srcs = ["origin_info_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct/testing",
    ],
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "pretty_printer_test",
    srcs = ["pretty_printer_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "qual_names_test",
    srcs = ["qual_names_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "templates_test",
    srcs = ["templates_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "transformer_test",
    srcs = ["transformer_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pyct",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)

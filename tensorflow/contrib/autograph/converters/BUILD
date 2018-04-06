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
    name = "converters",
    srcs = [
        "asserts.py",
        "break_statements.py",
        "builtin_functions.py",
        "call_trees.py",
        "continue_statements.py",
        "control_flow.py",
        "decorators.py",
        "ifexp.py",
        "list_comprehension.py",
        "lists.py",
        "logical_expressions.py",
        "name_scopes.py",
        "side_effect_guards.py",
        "single_return.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "@gast_archive//:gast",
    ],
)

py_library(
    name = "test_lib",
    srcs = [
        "converter_test_base.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":converters",
        "//tensorflow/contrib/autograph/operators",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/contrib/autograph/pyct/static_analysis",
        "//tensorflow/contrib/autograph/utils",
        "@gast_archive//:gast",
        "@six_archive//:six",
    ],
)

py_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "break_statements_test",
    srcs = ["break_statements_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "builtin_functions_test",
    srcs = ["builtin_functions_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "call_trees_test",
    size = "large",
    srcs = ["call_trees_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":test_lib",
        "//tensorflow/contrib/autograph/impl",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "continue_statements_test",
    srcs = ["continue_statements_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "decorators_test",
    srcs = ["decorators_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "name_scopes_test",
    srcs = ["name_scopes_test.py"],
    deps = [
        ":test_lib",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "list_comprehension_test",
    srcs = ["list_comprehension_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "lists_test",
    srcs = ["lists_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "logical_expressions_test",
    srcs = ["logical_expressions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "side_effect_guards_test",
    srcs = ["side_effect_guards_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        # TODO(mdan): Fix.
        "flaky",
        "notap",
    ],
    deps = [
        ":test_lib",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "single_return_test",
    srcs = ["single_return_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "ifexp_test",
    srcs = ["ifexp_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_lib",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/python:client_testlib",
    ],
)

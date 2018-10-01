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
        "conditional_expressions.py",
        "continue_statements.py",
        "control_flow.py",
        "decorators.py",
        "directives.py",
        "error_handlers.py",
        "list_comprehensions.py",
        "lists.py",
        "logical_expressions.py",
        "name_scopes.py",
        "return_statements.py",
        "side_effect_guards.py",
        "slices.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/lang",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "break_statements_test",
    srcs = ["break_statements_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "builtin_functions_test",
    srcs = ["builtin_functions_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "call_trees_test",
    size = "large",
    srcs = ["call_trees_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/impl",
    ],
)

py_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "continue_statements_test",
    srcs = ["continue_statements_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "decorators_test",
    srcs = ["decorators_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "directives_test",
    srcs = ["directives_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang",
    ],
)

py_test(
    name = "name_scopes_test",
    srcs = ["name_scopes_test.py"],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_test(
    name = "list_comprehensions_test",
    srcs = ["list_comprehensions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "lists_test",
    srcs = ["lists_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "logical_expressions_test",
    srcs = ["logical_expressions_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "side_effect_guards_test",
    srcs = ["side_effect_guards_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "return_statements_test",
    srcs = ["return_statements_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_test(
    name = "error_handlers_test",
    srcs = ["error_handlers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
    ],
)

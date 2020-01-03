load("//tensorflow:tensorflow.bzl", "py_test")

package(
    licenses = ["notice"],  # Apache 2.0
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
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "converters",
    srcs = [
        "arg_defaults.py",
        "asserts.py",
        "break_statements.py",
        "call_trees.py",
        "conditional_expressions.py",
        "continue_statements.py",
        "control_flow.py",
        "directives.py",
        "function_scopes.py",
        "list_comprehensions.py",
        "lists.py",
        "logical_expressions.py",
        "return_statements.py",
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
        "//tensorflow/python/autograph/utils",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "arg_defaults_test",
    srcs = ["arg_defaults_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "break_statements_test",
    srcs = ["break_statements_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "call_trees_test",
    srcs = ["call_trees_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_pip",
        "no_windows",
        "nopip",
    ],
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        # TOODO(b/145618471): Remove this transitive dependency.
        "//tensorflow/python/distribute:input_lib",
    ],
)

py_test(
    name = "directives_test",
    srcs = ["directives_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang",
    ],
)

py_test(
    name = "function_scopes_test",
    srcs = ["function_scopes_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_test(
    name = "return_statements_test",
    srcs = ["return_statements_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":converters",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
    ],
)

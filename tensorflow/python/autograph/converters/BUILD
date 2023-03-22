load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":list_comprehensions",
    ],
)

py_strict_library(
    name = "slices",
    srcs = ["slices.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "continue_statements",
    srcs = ["continue_statements.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
    ],
)

py_strict_library(
    name = "list_comprehensions",
    srcs = ["list_comprehensions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "logical_expressions",
    srcs = ["logical_expressions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "break_statements",
    srcs = ["break_statements.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
    ],
)

py_strict_library(
    name = "conditional_expressions",
    srcs = ["conditional_expressions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "call_trees",
    srcs = ["call_trees.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/utils:ag_logging",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "control_flow_deprecated_py2",
    srcs = ["control_flow_deprecated_py2.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "//tensorflow/python/autograph/pyct/static_analysis:liveness",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_definitions",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_fndefs",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "directives",
    srcs = ["directives.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/util:tf_inspect",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "variables",
    srcs = ["variables.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "//tensorflow/python/autograph/pyct/static_analysis:liveness",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_definitions",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_fndefs",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "functions",
    srcs = ["functions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "lists",
    srcs = ["lists.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "return_statements",
    srcs = ["return_statements.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "asserts",
    srcs = ["asserts.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":asserts",
        ":functions",
        ":return_statements",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
    ],
)

py_strict_test(
    name = "break_statements_test",
    srcs = ["break_statements_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":break_statements",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_strict_test(
    name = "call_trees_test",
    srcs = ["call_trees_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":call_trees",
        ":functions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_strict_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conditional_expressions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_strict_test(
    name = "continue_statements_test",
    srcs = ["continue_statements_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":continue_statements",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":break_statements",
        ":continue_statements",
        ":control_flow",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "directives_test",
    srcs = ["directives_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":directives",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_strict_test(
    name = "functions_test",
    srcs = ["functions_test.py"],
    python_version = "PY3",
    deps = [
        ":functions",
        ":return_statements",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/framework:constant_op",
    ],
)

py_strict_test(
    name = "list_comprehensions_test",
    srcs = ["list_comprehensions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":list_comprehensions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

py_strict_test(
    name = "lists_test",
    srcs = ["lists_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":directives",
        ":lists",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_test(
    name = "logical_expressions_test",
    srcs = ["logical_expressions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logical_expressions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "return_statements_test",
    srcs = ["return_statements_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":functions",
        ":return_statements",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":directives",
        ":slices",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
    ],
)

py_strict_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":variables",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:test_lib",
    ],
)

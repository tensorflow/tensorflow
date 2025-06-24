load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "slices",
    srcs = ["slices.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "continue_statements",
    srcs = ["continue_statements.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
    ],
)

py_strict_library(
    name = "logical_expressions",
    srcs = ["logical_expressions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "break_statements",
    srcs = ["break_statements.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
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
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "call_trees",
    srcs = ["call_trees.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/utils:ag_logging",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "directives",
    srcs = ["directives.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/util:tf_inspect",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "variables",
    srcs = ["variables.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:origin_info",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "//tensorflow/python/autograph/pyct/static_analysis:liveness",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_definitions",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_fndefs",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "functions",
    srcs = ["functions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "lists",
    srcs = ["lists.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "return_statements",
    srcs = ["return_statements.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:annos",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "asserts",
    srcs = ["asserts.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    deps = [
        ":asserts",
        ":functions",
        ":return_statements",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "break_statements_test",
    srcs = ["break_statements_test.py"],
    deps = [
        ":break_statements",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "call_trees_test",
    srcs = ["call_trees_test.py"],
    deps = [
        ":call_trees",
        ":functions",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    deps = [
        ":conditional_expressions",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "continue_statements_test",
    srcs = ["continue_statements_test.py"],
    deps = [
        ":continue_statements",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    deps = [
        ":break_statements",
        ":continue_statements",
        ":control_flow",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_test(
    name = "directives_test",
    srcs = ["directives_test.py"],
    deps = [
        ":directives",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "functions_test",
    srcs = ["functions_test.py"],
    deps = [
        ":functions",
        ":return_statements",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "lists_test",
    srcs = ["lists_test.py"],
    deps = [
        ":directives",
        ":lists",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "logical_expressions_test",
    srcs = ["logical_expressions_test.py"],
    deps = [
        ":logical_expressions",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "return_statements_test",
    srcs = ["return_statements_test.py"],
    deps = [
        ":functions",
        ":return_statements",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    deps = [
        ":directives",
        ":slices",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    deps = [
        ":variables",
        #internal proto upb dep
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

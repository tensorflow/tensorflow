load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "py_builtins",
    srcs = ["py_builtins.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops_gen",
        "//tensorflow/python:string_ops_gen",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/autograph/utils:tensors",
        "//tensorflow/python/autograph/utils:type_registry",
    ],
)

py_strict_library(
    name = "exceptions",
    srcs = ["exceptions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":conditional_expressions",
        ":control_flow",
        ":data_structures",
        ":exceptions",
        ":logical",
        ":py_builtins",
        ":slices",
        ":variables",
    ],
)

py_strict_library(
    name = "logical",
    srcs = ["logical.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:cond",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:tensor_util",
    ],
)

py_strict_library(
    name = "variables",
    srcs = ["variables.py"],
    visibility = ["//tensorflow:__subpackages__"],
)

py_strict_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
    ],
)

py_strict_library(
    name = "conditional_expressions",
    srcs = ["conditional_expressions.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":control_flow",
        "//tensorflow/python:cond",
        "//tensorflow/python/autograph/utils:tensors",
    ],
)

py_strict_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":py_builtins",
        ":variables",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/autograph/utils:misc",
        "//tensorflow/python/autograph/utils:tensors",
        "//tensorflow/python/autograph/utils:type_registry",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "slices",
    srcs = ["slices.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:string_ops_gen",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
    ],
)

py_strict_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_structures",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
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
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_gpu",  # b/127001953
    ],
    deps = [
        ":control_flow",
        ":variables",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/autograph/utils:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "exceptions_test",
    srcs = ["exceptions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":exceptions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logical",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "py_builtins_test",
    srcs = ["py_builtins_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_structures",
        ":py_builtins",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:function_wrappers",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":slices",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:list_ops",
        "//tensorflow/python/framework:constant_op",
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
    ],
)

pytype_strict_library(
    name = "dispatch_context",
    srcs = ["dispatch_context.py"],
)

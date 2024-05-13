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
        "//tensorflow/python/autograph/utils:tensors",
        "//tensorflow/python/autograph/utils:type_registry",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:parsing_ops_gen",
        "//tensorflow/python/ops:string_ops_gen",
    ],
)

py_strict_library(
    name = "exceptions",
    srcs = ["exceptions.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:control_flow_assert",
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
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops_gen",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:tensor_array_ops",
    ],
)

py_strict_library(
    name = "conditional_expressions",
    srcs = ["conditional_expressions.py"],
    visibility = ["//visibility:private"],
    deps = [
        ":control_flow",
        "//tensorflow/python/autograph/utils:tensors",
        "//tensorflow/python/ops:cond",
    ],
)

py_strict_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":py_builtins",
        ":variables",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/autograph/utils:misc",
        "//tensorflow/python/autograph/utils:tensors",
        "//tensorflow/python/autograph/utils:type_registry",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "slices",
    srcs = ["slices.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:string_ops_gen",
        "//tensorflow/python/ops:tensor_array_ops",
    ],
)

py_strict_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_structures",
        #internal proto upb dep
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conditional_expressions",
        #internal proto upb dep
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
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
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/autograph/utils:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "exceptions_test",
    srcs = ["exceptions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":exceptions",
        #internal proto upb dep
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logical",
        #internal proto upb dep
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
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
        #internal proto upb dep
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:function_wrappers",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":slices",
        #internal proto upb dep
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":variables",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "dispatch_context",
    srcs = ["dispatch_context.py"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "operators",
    srcs = [
        "__init__.py",
        "conditional_expressions.py",
        "control_flow.py",
        "data_structures.py",
        "exceptions.py",
        "logical.py",
        "py_builtins.py",
        "slices.py",
        "variables.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/autograph/utils:misc",
        "//tensorflow/python/autograph/utils:tensors",
        "//tensorflow/python/autograph/utils:type_registry",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/experimental/ops:scan_ops",
        "//tensorflow/python/data/experimental/ops:take_while_ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_gpu",  # b/127001953
    ],
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/utils:testing",
        # TODO(b/145618471): Remove this transitive dependency.
        "//tensorflow/python/distribute:input_lib",
    ],
)

py_test(
    name = "exceptions_test",
    srcs = ["exceptions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "py_builtins_test",
    srcs = ["py_builtins_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:function_wrappers",
        # TODO(b/145618471): Remove this transitive dependency.
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
    ],
)

py_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":operators",
        "//tensorflow/python:client_testlib",
    ],
)

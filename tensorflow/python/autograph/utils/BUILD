load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "type_registry",
    srcs = [
        "type_registry.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
)

py_strict_library(
    name = "tensor_list",
    srcs = ["tensor_list.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "ag_logging",
    srcs = ["ag_logging.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "context_managers",
    srcs = ["context_managers.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "misc",
    srcs = ["misc.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "testing",
    srcs = ["testing.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:op_callbacks",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":context_managers",
        ":misc",
        ":tensor_list",
    ],
)

py_strict_library(
    name = "tensors",
    srcs = ["tensors.py"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_test(
    name = "context_managers_test",
    srcs = ["context_managers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_managers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/framework:constant_op",
    ],
)

py_strict_test(
    name = "misc_test",
    srcs = ["misc_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":misc",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "tensor_list_test",
    srcs = ["tensor_list_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor_list",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_test(
    name = "tensors_test",
    srcs = ["tensors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensors",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/framework:constant_op",
    ],
)

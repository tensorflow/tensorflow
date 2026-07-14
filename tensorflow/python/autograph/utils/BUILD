load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "type_registry",
    srcs = [
        "type_registry.py",
    ],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "tensor_list",
    srcs = ["tensor_list.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:tensor_array_ops",
    ],
)

py_library(
    name = "ag_logging",
    srcs = ["ag_logging.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "context_managers",
    srcs = ["context_managers.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:tensor_array_ops",
    ],
)

py_library(
    name = "misc",
    srcs = ["misc.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
    ],
)

py_library(
    name = "testing",
    srcs = ["testing.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:op_callbacks",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":context_managers",
        ":misc",
        ":tensor_list",
    ],
)

py_library(
    name = "tensors",
    srcs = ["tensors.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:tensor_array_ops",
    ],
)

py_test(
    name = "context_managers_test",
    srcs = ["context_managers_test.py"],
    strict_deps = True,
    deps = [
        ":context_managers",
        #internal proto upb dep
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "misc_test",
    srcs = ["misc_test.py"],
    strict_deps = True,
    deps = [
        ":misc",
        #internal proto upb dep
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "tensor_list_test",
    srcs = ["tensor_list_test.py"],
    strict_deps = True,
    deps = [
        ":tensor_list",
        #internal proto upb dep
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "tensors_test",
    srcs = ["tensors_test.py"],
    strict_deps = True,
    deps = [
        ":tensors",
        #internal proto upb dep
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

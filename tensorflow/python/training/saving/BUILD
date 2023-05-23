# Description:
#   Low-level utilities for reading and writing checkpoints.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/checkpoint:functional_saver",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "saveable_object",
    srcs = ["saveable_object.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "saveable_object_util",
    srcs = ["saveable_object_util.py"],
    srcs_version = "PY3",
    deps = [
        ":saveable_object",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops_gen",
        "//tensorflow/python:ref_variable",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "trace_saveable_util",
    srcs = ["trace_saveable_util.py"],
    deps = [
        ":saveable_object",
        ":saveable_object_util",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util:nest",
    ],
)

tf_py_strict_test(
    name = "saveable_object_util_test",
    srcs = ["saveable_object_util_test.py"],
    deps = [
        ":saveable_object",
        ":saveable_object_util",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
    ],
)

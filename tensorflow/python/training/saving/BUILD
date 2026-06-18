# Description:
#   Low-level utilities for reading and writing checkpoints.

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/util:deprecation",
    ],
)

py_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/checkpoint:functional_saver",
        "//tensorflow/python/util:deprecation",
    ],
)

py_library(
    name = "saveable_object",
    srcs = ["saveable_object.py"],
    strict_deps = True,
)

py_library(
    name = "saveable_object_util",
    srcs = ["saveable_object_util.py"],
    strict_deps = True,
    deps = [
        ":saveable_object",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops_gen",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:base_delegate",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "trace_saveable_util",
    srcs = ["trace_saveable_util.py"],
    strict_deps = True,
    deps = [
        ":saveable_object",
        ":saveable_object_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/util:nest",
    ],
)

tf_py_strict_test(
    name = "saveable_object_util_test",
    srcs = ["saveable_object_util_test.py"],
    deps = [
        ":saveable_object",
        ":saveable_object_util",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
    ],
)

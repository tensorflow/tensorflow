# Description:
#   Low-level utilities for reading and writing checkpoints.

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_options",
        ":saveable_object",
        ":saveable_object_util",
        "//tensorflow/python:util",
        "//tensorflow/python/checkpoint:functional_saver",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/saved_model/registration",
    ],
)

py_library(
    name = "saveable_object",
    srcs = ["saveable_object.py"],
    srcs_version = "PY3",
)

py_library(
    name = "saveable_object_util",
    srcs = ["saveable_object_util.py"],
    srcs_version = "PY3",
    deps = [
        ":saveable_object",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:python_state",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/types",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "saveable_object_util_test",
    srcs = ["saveable_object_util_test.py"],
    deps = [
        ":saveable_object_util",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
    ],
)

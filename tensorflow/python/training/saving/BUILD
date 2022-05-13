# Description:
#   Low-level utilities for reading and writing checkpoints.

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
        ":saveable_hook",
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
    name = "saveable_hook",
    srcs = ["saveable_hook.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python/training/tracking:base",
    ],
)

py_library(
    name = "saveable_object_util",
    srcs = ["saveable_object_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:trackable_utils",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

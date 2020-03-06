package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

#TODO(scottzhu): Cleanup all the deps to python/keras

py_library(
    name = "frozen_keras",
    deps = [
        ":constraint",
        ":initializers",
        ":regularizers",
        "//tensorflow/python/frozen_keras/engine:legacy_base_layer",
    ],
)

py_library(
    name = "constraint",
    srcs = ["constraints.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:generic_utils",
        "@six_archive//:six",
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:tf2",
        "//tensorflow/python/keras/utils:generic_utils",
        "@six_archive//:six",
    ],
)

py_library(
    name = "regularizers",
    srcs = ["regularizers.py"],
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:generic_utils",
        "@six_archive//:six",
    ],
)

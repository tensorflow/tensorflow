# Description:
#   Contains the legacy TF layers (internal TensorFlow version).

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
        "variable_scope_shim.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/mixed_precision:policy",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//tensorflow/python:init_ops",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//tensorflow/python:init_ops",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/util:tf_export",
    ],
)

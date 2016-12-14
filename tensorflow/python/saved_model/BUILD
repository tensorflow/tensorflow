# Description:
# TensorFlow SavedModel.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "signature_constants",
    srcs = ["signature_constants.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "tag_constants",
    srcs = ["tag_constants.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "builder",
    srcs = ["builder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "loader",
    srcs = ["loader.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "main_op",
    srcs = ["main_op.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "saved_model_test",
    size = "small",
    srcs = [
        "saved_model_test.py",
    ],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    visibility = ["//visibility:private"],
    deps = [
        ":builder",
        ":loader",
        ":main_op",
        ":tag_constants",
        ":utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = [
        "utils_test.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":utils",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "signature_def_utils",
    srcs = ["signature_def_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":signature_constants",
        ":utils",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "signature_def_utils_test",
    size = "small",
    srcs = [
        "signature_def_utils_test.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":signature_def_utils",
        "//tensorflow:tensorflow_py",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

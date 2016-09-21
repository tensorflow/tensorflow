# Description:
# TensorFlow SavedModel.

package(
    default_visibility = ["//tensorflow/python/saved_model:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "builder",
    srcs = ["builder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "saved_model_test",
    size = "small",
    srcs = [
        "saved_model_test.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":builder",
        ":loader",
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

# Description:
#   TensorFlow Model Summary CLI tool

load("//tensorflow:py.default.bzl", "py_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "model_loader",
    srcs = ["model_loader.py"],
    deps = [
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "layer_parser",
    srcs = ["layer_parser.py"],
    deps = [
        "//tensorflow/python/keras/utils:layer_utils",
    ],
)

py_strict_library(
    name = "formatter",
    srcs = ["formatter.py"],
)

py_strict_library(
    name = "visualization",
    srcs = ["visualization.py"],
    deps = [
        "//tensorflow/python/keras/utils:vis_utils",
    ],
)

py_strict_library(
    name = "cli",
    srcs = ["cli.py"],
    deps = [
        ":formatter",
        ":layer_parser",
        ":model_loader",
        ":visualization",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "model_loader_test",
    srcs = ["model_loader_test.py"],
    deps = [
        ":model_loader",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/keras:models",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "layer_parser_test",
    srcs = ["layer_parser_test.py"],
    deps = [
        ":layer_parser",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/keras:models",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "cli_test",
    srcs = ["cli_test.py"],
    deps = [
        ":cli",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/keras:models",
        "//tensorflow/python/platform:test",
    ],
)

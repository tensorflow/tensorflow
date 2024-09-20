load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(command_line_flags)

cc_library(
    name = "command_line_flags",
    srcs = ["command_line_flags.cc"],
    hdrs = ["command_line_flags.h"],
    copts = tflite_copts(),
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

# LINT.ThenChange(//tensorflow/lite/tools:command_line_flags)

py_strict_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/mlir/lite/python:schema_py",
        "//tensorflow/compiler/mlir/lite/python:schema_util",
        "//tensorflow/python/platform:gfile",
        "@flatbuffers//:runtime_py",
    ],
)

py_strict_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flatbuffer_utils",
        ":test_utils",
        #internal proto upb dep
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/mlir/lite/python:schema_py",
        "@flatbuffers//:runtime_py",
    ],
)

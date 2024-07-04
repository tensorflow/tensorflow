load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "logging",
    hdrs = ["//tensorflow/lite/tools:logging.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
)

cc_library(
    name = "command_line_flags",
    srcs = ["command_line_flags.cc"],
    hdrs = ["command_line_flags.h"],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        "@com_google_absl//absl/strings",
    ],
)

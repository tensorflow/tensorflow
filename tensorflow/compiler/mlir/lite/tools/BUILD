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

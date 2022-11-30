# Tools to strip out eligible buffers (tensor data) from a TFLite flatbuffer, and load them with
# random values at runtime.

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "stripping_lib",
    srcs = ["stripping_lib.cc"],
    hdrs = ["stripping_lib.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "strip_buffers_from_fb",
    srcs = ["strip_buffers_from_fb.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        ":stripping_lib",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools:command_line_flags",
    ],
)

cc_binary(
    name = "reconstitute_buffers_into_fb",
    srcs = ["reconstitute_buffers_into_fb.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        ":stripping_lib",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools:command_line_flags",
    ],
)

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "android_required_build_flags",
    srcs = ["android_required_build_flags.cc"],
    copts = tflite_copts(),
)

cc_library(
    name = "csv_writer",
    hdrs = ["csv_writer.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

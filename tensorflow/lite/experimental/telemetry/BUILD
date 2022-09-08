load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_settings",
        ":telemetry_status",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry.cc"],
    hdrs = ["telemetry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_settings",
        ":telemetry_status",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    deps = [
        ":profiler",
        ":telemetry",
        ":telemetry_status",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "telemetry_status",
    hdrs = ["telemetry_status.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = ["//tensorflow/lite/c:c_api_types"],
)

cc_library(
    name = "telemetry_settings",
    hdrs = ["telemetry_settings.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [":telemetry_status"],
)

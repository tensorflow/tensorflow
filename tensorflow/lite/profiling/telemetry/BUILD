load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        ":telemetry_status",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/profiling/telemetry/c:profiler",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
    ],
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry.cc"],
    hdrs = ["telemetry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_status",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
    ],
)

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    deps = [
        ":profiler",
        ":telemetry",
        ":telemetry_status",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting_internal",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    deps = [
        ":profiler",
        ":telemetry",
        ":telemetry_status",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "telemetry_status",
    hdrs = ["telemetry_status.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = ["//tensorflow/lite/core/c:c_api_types"],
)

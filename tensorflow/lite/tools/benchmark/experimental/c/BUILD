load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["benchmark"],
    licenses = ["notice"],
)

package_group(
    name = "benchmark",
    packages = [
        "//tensorflow/lite/tools/benchmark/...",
    ],
)

exports_files(
    ["benchmark_c_api.h"],
    visibility = ["//tensorflow/lite/tools/benchmark/experimental/c:benchmark"],
)

cc_library(
    name = "benchmark_c_api",
    srcs = ["benchmark_c_api.cc"],
    hdrs = [
        "benchmark_c_api.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "benchmark",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/tools/benchmark:benchmark_model_lib",
        "//tensorflow/lite/tools/benchmark:benchmark_params",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
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
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)

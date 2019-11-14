load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["benchmark"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "benchmark",
    packages = [
        "//tensorflow/lite/tools/benchmark/...",
    ],
)

cc_library(
    name = "benchmark_c_api",
    srcs = ["benchmark_c_api.cc"],
    hdrs = [
        "benchmark_c_api.h",
        "c_api_types.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "benchmark",
    ],
    deps = [
        "//tensorflow/core:stats_calculator_portable",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)

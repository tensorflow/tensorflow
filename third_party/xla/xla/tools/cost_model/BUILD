load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_bandwidth_benchmark",
    srcs = ["gpu_bandwidth_benchmark.cc"],
    hdrs = ["gpu_bandwidth_benchmark.h"],
    deps = [
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_init",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ] + if_cuda([
        "//xla/stream_executor/cuda:cuda_platform",
    ]),
)

xla_cc_test(
    name = "gpu_bandwidth_benchmark_test",
    srcs = ["gpu_bandwidth_benchmark_test.cc"],
    tags = [
        "cuda-only",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":gpu_bandwidth_benchmark",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
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
        "//xla/stream_executor/gpu:gpu_init",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ] + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
)

xla_test(
    name = "gpu_bandwidth_benchmark_test",
    srcs = ["gpu_bandwidth_benchmark_test.cc"],
    backends = [
        "h100",
        "amdgpu_any",
    ],
    deps = [
        ":gpu_bandwidth_benchmark",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

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
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "gpu_bandwidth_benchmark_test",
    srcs = ["gpu_bandwidth_benchmark_test.cc"],
    deps = [
        ":gpu_bandwidth_benchmark",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

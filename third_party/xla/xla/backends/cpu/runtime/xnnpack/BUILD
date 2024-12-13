load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "object_pool",
    hdrs = ["object_pool.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
    ],
)

xla_cc_test(
    name = "object_pool_test",
    srcs = ["object_pool_test.cc"],
    deps = [
        ":object_pool",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "parallel_loop_runner",
    srcs = ["parallel_loop_runner.cc"],
    hdrs = ["parallel_loop_runner.h"],
    deps = [
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/math:math_util",
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "parallel_loop_runner_test",
    srcs = ["parallel_loop_runner_test.cc"],
    deps = [
        ":parallel_loop_runner",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "xnn_interop",
    hdrs = ["xnn_interop.h"],
    deps = [
        "@XNNPACK",
    ],
)

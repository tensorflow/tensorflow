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

cc_library(
    name = "xnn_threadpool",
    srcs = ["xnn_threadpool.cc"],
    hdrs = ["xnn_threadpool.h"],
    # copybara:uncomment_begin(google-only)
    # local_defines = select({
    # "@pthreadpool:pthreadpool_header_only_explicit_true": [
    # "XLA_CPU_USE_CUSTOM_PTHREADPOOL",
    # ],
    # "//conditions:default": [],
    # }),
    # copybara:uncomment_end
    deps = [
        ":parallel_loop_runner",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@pthreadpool",
    ],
)

xla_cc_test(
    name = "xnn_threadpool_test",
    srcs = ["xnn_threadpool_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":parallel_loop_runner",
        ":xnn_threadpool",
        "//xla/tsl/concurrency:async_value",
        "@XNNPACK",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
        "@pthreadpool",
    ],
)

load("@local_tsl//tsl/platform:build_config.bzl", "tsl_cc_test")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "async_value",
    srcs = [
        "async_value.cc",
        "async_value_ref.cc",
    ],
    hdrs = [
        "async_value.h",
        "async_value_ref.h",
        "chain.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":concurrent_vector",
        ":ref_count",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

tsl_cc_test(
    name = "async_value_test",
    srcs = ["async_value_test.cc"],
    deps = [
        ":async_value",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "async_value_ptr_test",
    srcs = ["async_value_ptr_test.cc"],
    deps = [
        ":async_value",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "async_value_ref_test",
    srcs = ["async_value_ref_test.cc"],
    deps = [
        ":async_value",
        ":ref_count",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "concurrent_vector",
    hdrs = ["concurrent_vector.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

tsl_cc_test(
    name = "concurrent_vector_test",
    srcs = ["concurrent_vector_test.cc"],
    deps = [
        ":concurrent_vector",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "ref_count",
    hdrs = ["ref_count.h"],
    compatible_with = get_compatible_with_portable(),
)

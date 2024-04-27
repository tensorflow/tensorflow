load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "async_value",
    hdrs = [
        "async_value.h",
        "async_value_ref.h",
        "chain.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":concurrent_vector",
        ":ref_count",
        "//tsl/platform:logging",
        "//tsl/platform:platform_port",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/concurrency:async_value",
    ],
)

cc_library(
    name = "concurrent_vector",
    hdrs = ["concurrent_vector.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tsl/platform:logging",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/concurrency:concurrent_vector",
    ],
)

cc_library(
    name = "ref_count",
    hdrs = ["ref_count.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/concurrency:ref_count",
    ],
)

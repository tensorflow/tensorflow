load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/runtime:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "async_values_cache",
    hdrs = ["async_values_cache.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "memory_mapper",
    hdrs = ["memory_mapper.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [],
)

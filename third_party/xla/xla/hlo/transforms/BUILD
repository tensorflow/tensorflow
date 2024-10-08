# Description:
#   Implementation of XLA’s HLO transformations.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "hlo_constant_splitter",
    srcs = ["hlo_constant_splitter.cc"],
    hdrs = ["hlo_constant_splitter.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_constant_splitter_test",
    srcs = ["hlo_constant_splitter_test.cc"],
    deps = [
        ":hlo_constant_splitter",
        "//xla:test",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/test_utils:hlo_hardware_independent_test_base",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_parser",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

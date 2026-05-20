load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Description:
#   A tool for reducing a HLO module that produces incorrect results.
load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
)

cc_library(
    name = "hlo_bisect_state",
    srcs = ["hlo_bisect_state.cc"],
    hdrs = ["hlo_bisect_state.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_bisect_state_test",
    srcs = ["hlo_bisect_state_test.cc"],
    deps = [
        ":hlo_bisect_state",
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/service:pattern_matcher",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
    ],
)

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_gumgraph_node",
    hdrs = ["hlo_gumgraph_node.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "hlo_gumgraph",
    srcs = ["hlo_gumgraph.cc"],
    hdrs = ["hlo_gumgraph.h"],
    deps = [
        ":hlo_gumgraph_node",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff/graph/analysis:hlo_value_tracing",
        "//xla/hlo/tools/hlo_diff/graph/utils:cycle_detector",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_dfs",
        "//xla/hlo/tools/hlo_diff/utils:hlo_diff_util",
        "//xla/service:call_graph",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_test",
    srcs = ["hlo_gumgraph_test.cc"],
    deps = [
        ":hlo_gumgraph",
        ":hlo_gumgraph_node",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

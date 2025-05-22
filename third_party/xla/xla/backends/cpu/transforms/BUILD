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
    name = "xnn_graph_fusion",
    srcs = ["xnn_graph_fusion.cc"],
    hdrs = ["xnn_graph_fusion.h"],
    deps = [
        "//xla/backends/cpu:xnn_fusion",
        "//xla/backends/cpu:xnnpack_config_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:instruction_fusion",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "xnn_graph_fusion_test",
    srcs = ["xnn_graph_fusion_test.cc"],
    deps = [
        ":xnn_graph_fusion",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:xnn_fusion",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest",
    ],
)

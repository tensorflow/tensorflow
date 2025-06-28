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
    name = "dot_library_rewriter",
    srcs = ["dot_library_rewriter.cc"],
    hdrs = ["dot_library_rewriter.h"],
    deps = [
        ":library_matcher",
        ":onednn_matcher",
        ":xnn_matcher",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "dot_library_rewriter_test",
    srcs = ["dot_library_rewriter_test.cc"],
    deps = [
        ":dot_library_rewriter",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/backends/cpu/codegen:target_machine_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "library_matcher",
    hdrs = ["library_matcher.h"],
    deps = [
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "onednn_matcher",
    hdrs = ["onednn_matcher.h"],
    deps = [
        ":library_matcher",
        "//xla/backends/cpu:onednn_fusion",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "xnn_matcher",
    hdrs = ["xnn_matcher.h"],
    deps = [
        ":library_matcher",
        "//xla/backends/cpu:xnn_fusion",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
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

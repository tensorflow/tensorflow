load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/mkl:build_defs.bzl", "if_graph_api")
load("//xla/tsl/mkl:graph.bzl", "onednn_graph_cc_library")
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
    name = "library_rewriter",
    srcs = ["library_rewriter.cc"],
    hdrs = ["library_rewriter.h"],
    defines = if_graph_api(["XLA_ONEDNN_USE_GRAPH_API=1"]),
    deps = [
        ":library_matcher",
        ":onednn_matcher",
        ":ynn_matcher",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf_lite",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "library_rewriter_test",
    srcs = ["library_rewriter_test.cc"],
    local_defines = if_graph_api(["XLA_ONEDNN_USE_GRAPH_API=1"]),
    deps = [
        ":library_rewriter",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/backends/cpu/codegen:target_machine_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:no_destructor",
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
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:protobuf",
    ],
)

onednn_graph_cc_library(
    name = "onednn_matcher",
    hdrs = ["onednn_matcher.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":library_matcher",
        "//xla/backends/cpu:onednn_support",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "ynn_matcher",
    hdrs = ["ynn_matcher.h"],
    deps = [
        ":library_matcher",
        "//xla/backends/cpu:ynn_support",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "ynn_matcher_test",
    srcs = ["ynn_matcher_test.cc"],
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/service:cpu_plugin",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

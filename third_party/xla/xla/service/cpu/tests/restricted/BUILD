load("//xla:restricted_package.bzl", "xla_restricted_allow", "xla_restricted_verify")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "tsl_copts")
load("//xla/tsl/mkl:build_defs.bzl", "if_graph_api")
load("//xla/tsl/mkl:graph.bzl", "onednn_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/cpu/tests:friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

xla_cc_test(
    name = "cpu_aot_export_test",
    srcs = ["cpu_aot_export_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

xla_cc_test(
    name = "onednn_fusion_test",
    srcs = ["onednn_fusion_test.cc"],
    local_defines = if_graph_api(["XLA_ONEDNN_USE_GRAPH_API=1"]),
    deps = [
        "//xla:error_spec",
        "//xla/service:cpu_plugin",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:platform_port",
    ],
)

onednn_cc_test(
    name = "onednn_matmul_test",
    srcs = ["onednn_matmul_test.cc"],
    copts = tsl_copts(),
    shard_count = 4,
    tags = [
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:error_spec",
        "//xla/hlo/testlib:test",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "@com_google_absl//absl/strings",
    ],
)

onednn_cc_test(
    name = "onednn_convolution_test",
    srcs = ["onednn_convolution_test.cc"],
    copts = tsl_copts(),
    tags = [
        # TODO: reenable once onednn is supported by the thunk runtime.
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_contraction_rewriter",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

onednn_cc_test(
    name = "onednn_layer_norm_test",
    srcs = ["onednn_layer_norm_test.cc"],
    copts = tsl_copts(),
    tags = [
        # TODO: reenable once onednn is supported by the thunk runtime.
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:error_spec",
        "//xla/hlo/testlib:test",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
    ],
)

onednn_cc_test(
    name = "onednn_softmax_test",
    srcs = ["onednn_softmax_test.cc"],
    copts = tsl_copts(),
    shard_count = if_graph_api(4, 1),
    tags = [
        # TODO: reenable once onednn is supported by the thunk runtime.
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/service:cpu_plugin",
        "//xla/service:pattern_matcher",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:onednn_config_proto_cc",
        "//xla/service/cpu:onednn_ops_rewriter",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("cpu_aot_export_test"),
    xla_restricted_allow("onednn_convolution_test"),
    xla_restricted_allow("onednn_fusion_test"),
    xla_restricted_allow("onednn_layer_norm_test"),
    xla_restricted_allow("onednn_matmul_test"),
    xla_restricted_allow("onednn_softmax_test"),
])

load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")
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
    name = "hlo_gumgraph_mappings",
    hdrs = ["hlo_gumgraph_mappings.h"],
    deps = [
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/service:call_graph",
        "@boost//:bimap",
    ],
)

cc_library(
    name = "hlo_diff_result",
    srcs = ["hlo_diff_result.cc"],
    hdrs = ["hlo_diff_result.h"],
    deps = [
        ":hlo_gumgraph_mappings",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_bfs",
        "//xla/hlo/tools/hlo_diff/utils:hlo_diff_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "hlo_diff_result_test",
    srcs = ["hlo_diff_result_test.cc"],
    deps = [
        ":hlo_diff_result",
        ":hlo_gumgraph_mappings",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_diff_summary",
    srcs = ["hlo_diff_summary.cc"],
    hdrs = ["hlo_diff_summary.h"],
    deps = [
        ":hlo_diff_result",
        ":hlo_gumgraph_mappings",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/utils:connected_components",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "hlo_diff_summary_test",
    srcs = ["hlo_diff_summary_test.cc"],
    deps = [
        ":hlo_diff_result",
        ":hlo_diff_summary",
        ":hlo_gumgraph_mappings",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_gumgraph_diff",
    srcs = ["hlo_gumgraph_diff.cc"],
    hdrs = ["hlo_gumgraph_diff.h"],
    deps = [
        ":hlo_diff_eval",
        ":hlo_diff_result",
        ":hlo_diff_summary",
        ":hlo_gumgraph_mappings",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/matchers:hlo_call_graph_matcher",
        "//xla/hlo/tools/hlo_diff/matchers:hlo_computation_graph_matcher",
        "//xla/hlo/tools/hlo_diff/matchers:hlo_gumgraph_matcher",
        "//xla/service:call_graph",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_diff_test",
    srcs = ["hlo_gumgraph_diff_test.cc"],
    deps = [
        ":hlo_gumgraph_diff",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_diff_eval",
    srcs = ["hlo_diff_eval.cc"],
    hdrs = ["hlo_diff_eval.h"],
    deps = [
        ":hlo_diff_result",
        ":hlo_diff_summary",
        ":hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
    ],
)

xla_cc_test(
    name = "hlo_diff_eval_test",
    srcs = ["hlo_diff_eval_test.cc"],
    deps = [
        ":hlo_diff_eval",
        ":hlo_diff_result",
        ":hlo_diff_summary",
        ":hlo_gumgraph_mappings",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_binary(
    name = "hlo_diff",
    srcs = ["hlo_diff_main.cc"],
    deps = [
        ":hlo_diff_result",
        ":hlo_diff_summary",
        ":hlo_gumgraph_diff",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/tools/hlo_diff/render:hlo_gumgraph_html_renderer",
        "//xla/hlo/tools/hlo_diff/render:hlo_gumgraph_text_renderer",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

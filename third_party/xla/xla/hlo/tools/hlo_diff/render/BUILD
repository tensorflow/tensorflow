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
    name = "hlo_gumgraph_renderer_util",
    srcs = ["hlo_gumgraph_renderer_util.cc"],
    hdrs = ["hlo_gumgraph_renderer_util.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_renderer_util_test",
    srcs = ["hlo_gumgraph_renderer_util_test.cc"],
    deps = [
        ":hlo_gumgraph_renderer_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_gumgraph_text_renderer",
    srcs = ["hlo_gumgraph_text_renderer.cc"],
    hdrs = ["hlo_gumgraph_text_renderer.h"],
    deps = [
        ":hlo_gumgraph_renderer_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_gumgraph_html_renderer",
    srcs = ["hlo_gumgraph_html_renderer.cc"],
    hdrs = ["hlo_gumgraph_html_renderer.h"],
    deps = [
        ":graph_url_generator",
        ":hlo_gumgraph_renderer_util",
        ":op_metric_getter",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff:hlo_diff_summary",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_html_renderer_test",
    srcs = ["hlo_gumgraph_html_renderer_test.cc"],
    deps = [
        ":hlo_gumgraph_html_renderer",
        ":op_metric_getter",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff:hlo_diff_summary",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "graph_url_generator",
    hdrs = ["graph_url_generator.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "op_metric_getter",
    hdrs = ["op_metric_getter.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

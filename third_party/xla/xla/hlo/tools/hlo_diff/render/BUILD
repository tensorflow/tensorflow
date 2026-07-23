# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
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
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
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
        "//xla:printer",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff:hlo_diff_summary",
        "//xla/hlo/tools/hlo_diff/utils:text_diff",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
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

cc_library(
    name = "hlo_gumgraph_model_explorer_renderer",
    srcs = ["hlo_gumgraph_model_explorer_renderer.cc"],
    hdrs = ["hlo_gumgraph_model_explorer_renderer.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":hlo_gumgraph_renderer_util",
        "//third_party/json",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff:hlo_diff_summary",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:direct_hlo_to_json_graph_convert",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:schema_structs",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_model_explorer_renderer_test",
    srcs = ["hlo_gumgraph_model_explorer_renderer_test.cc"],
    deps = [
        ":hlo_gumgraph_model_explorer_renderer",
        "//third_party/json",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff:hlo_diff_summary",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/proto:diff_result_proto_cc",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:direct_hlo_to_json_graph_convert",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:schema_structs",
        "//xla/hlo/tools/hlo_diff/utils:hlo_diff_util",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_explorer_url_generator",
    srcs = ["model_explorer_url_generator.cc"],
    hdrs = ["model_explorer_url_generator.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":graph_url_generator",
        ":hlo_gumgraph_model_explorer_renderer",
        "//third_party/json",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:direct_hlo_to_json_graph_convert",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:schema_structs",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "model_explorer_url_generator_test",
    srcs = ["model_explorer_url_generator_test.cc"],
    deps = [
        ":hlo_gumgraph_model_explorer_renderer",
        ":model_explorer_url_generator",
        "//third_party/json",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_diff_result",
        "//xla/hlo/tools/hlo_diff/proto:diff_result_proto_cc",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:direct_hlo_to_json_graph_convert",
        "//xla/hlo/tools/hlo_diff/render/hlo_adapter:schema_structs",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)

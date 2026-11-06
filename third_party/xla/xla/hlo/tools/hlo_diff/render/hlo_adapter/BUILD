load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "direct_hlo_to_json_graph_convert",
    srcs = ["direct_hlo_to_json_graph_convert.cc"],
    hdrs = ["direct_hlo_to_json_graph_convert.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":graphnode_builder",
        ":schema_structs",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "graphnode_builder",
    srcs = ["graphnode_builder.cc"],
    hdrs = ["graphnode_builder.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":schema_structs",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "schema_structs",
    srcs = ["schema_structs.cc"],
    hdrs = ["schema_structs.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "direct_hlo_to_json_graph_convert_test",
    srcs = ["direct_hlo_to_json_graph_convert_test.cc"],
    deps = [
        ":direct_hlo_to_json_graph_convert",
        ":schema_structs",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

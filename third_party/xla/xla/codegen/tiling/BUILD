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
    name = "tiled_hlo_instruction",
    srcs = [
        "tiled_hlo_instruction.cc",
    ],
    hdrs = [
        "tiled_hlo_instruction.h",
    ],
    deps = [
        "//xla:util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "tiled_hlo_instruction_test",
    srcs = ["tiled_hlo_instruction_test.cc"],
    deps = [
        ":tiled_hlo_instruction",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tiled_hlo_computation",
    srcs = [
        "tiled_hlo_computation.cc",
    ],
    hdrs = [
        "tiled_hlo_computation.h",
    ],
    deps = [
        ":tiled_hlo_instruction",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:name_uniquer",
        "//xla/tsl/lib/gtl:iterator_range",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tiled_hlo_fusion_instruction",
    srcs = ["tiled_hlo_fusion_instruction.cc"],
    hdrs = ["tiled_hlo_fusion_instruction.h"],
    deps = [
        ":tiled_hlo_computation",
        ":tiled_hlo_instruction",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "tiled_hlo_fusion_instruction_test",
    srcs = ["tiled_hlo_fusion_instruction_test.cc"],
    deps = [
        ":tiled_hlo_fusion_instruction",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

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
        "//xla/tsl/platform:errors",
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
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
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
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tiled_hlo_schedule",
    srcs = ["tiled_hlo_schedule.cc"],
    hdrs = ["tiled_hlo_schedule.h"],
    deps = [
        ":tiling_specification",
        "//xla:util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "tiled_hlo_schedule_test",
    srcs = ["tiled_hlo_schedule_test.cc"],
    deps = [
        ":symbolic_tile_analysis",
        ":tiled_hlo_schedule",
        ":tiling_specification",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "affine_map_evaluator",
    srcs = ["affine_map_evaluator.cc"],
    hdrs = ["affine_map_evaluator.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "affine_map_evaluator_test",
    srcs = ["affine_map_evaluator_test.cc"],
    deps = [
        ":affine_map_evaluator",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "constraint_expression",
    srcs = ["constraint_expression.cc"],
    hdrs = ["constraint_expression.h"],
    deps = [
        ":affine_map_evaluator",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "constraint_expression_test",
    srcs = ["constraint_expression_test.cc"],
    deps = [
        ":constraint_expression",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "size_and_stride_expression",
    srcs = ["size_and_stride_expression.cc"],
    hdrs = ["size_and_stride_expression.h"],
    deps = [
        ":constraint_expression",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "symbolic_tile",
    srcs = ["symbolic_tile.cc"],
    hdrs = ["symbolic_tile.h"],
    deps = [
        ":affine_map_evaluator",
        ":constraint_expression",
        ":size_and_stride_expression",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tile_test",
    srcs = ["symbolic_tile_test.cc"],
    deps = [
        ":affine_map_evaluator",
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "symbolic_tiled_hlo_instruction",
    srcs = ["symbolic_tiled_hlo_instruction.cc"],
    hdrs = ["symbolic_tiled_hlo_instruction.h"],
    deps = [
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tiled_hlo_instruction_test",
    srcs = ["symbolic_tiled_hlo_instruction_test.cc"],
    deps = [
        ":symbolic_tile",
        ":symbolic_tiled_hlo_instruction",
        "//xla:shape_util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tiling_specification",
    srcs = ["tiling_specification.cc"],
    hdrs = ["tiling_specification.h"],
    deps = [
        ":constraint_expression",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "tiling_specification_test",
    srcs = ["tiling_specification_test.cc"],
    deps = [
        ":symbolic_tile_analysis",
        ":tiling_specification",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "symbolic_tile_analysis",
    srcs = ["symbolic_tile_analysis.cc"],
    hdrs = ["symbolic_tile_analysis.h"],
    deps = [
        ":constraint_expression",
        ":symbolic_tile",
        ":symbolic_tiled_hlo_instruction",
        ":tiled_hlo_computation",
        ":tiled_hlo_fusion_instruction",
        ":tiled_hlo_instruction",
        ":tiled_hlo_schedule",
        ":tiling_specification",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/service:name_uniquer",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:hash_container_defaults",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tile_analysis_test",
    srcs = ["symbolic_tile_analysis_test.cc"],
    deps = [
        ":constraint_expression",
        ":symbolic_tile_analysis",
        ":symbolic_tiled_hlo_instruction",
        ":tiled_hlo_computation",
        ":tiled_hlo_fusion_instruction",
        ":tiled_hlo_instruction",
        ":tiled_hlo_schedule",
        ":tiling_specification",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu/model:triton_emitter_constraints",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

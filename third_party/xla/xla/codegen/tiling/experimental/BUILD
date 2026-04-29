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
    name = "tile",
    srcs = [
        "tile.cc",
        "tiling_space.cc",
    ],
    hdrs = [
        "tile.h",
        "tiling_space.h",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/tiling:constraint_expression",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "tile_test",
    srcs = ["tile_test.cc"],
    deps = [
        ":tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "reshape_analysis",
    srcs = ["reshape_analysis.cc"],
    hdrs = ["reshape_analysis.h"],
    deps = [
        "//xla:shape_util",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "reshape_analysis_test",
    srcs = ["reshape_analysis_test.cc"],
    deps = [
        ":reshape_analysis",
        "//xla:shape_util",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tile_propagation",
    srcs = ["tile_propagation.cc"],
    hdrs = ["tile_propagation.h"],
    deps = [
        ":reshape_analysis",
        ":tile",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/service:matmul_indexing_utils",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "tile_propagation_test",
    srcs = ["tile_propagation_test.cc"],
    deps = [
        ":test_utils",
        ":tile",
        ":tile_propagation",
        "//xla:shape_util",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "tiling_space_test",
    srcs = ["tiling_space_test.cc"],
    deps = [
        ":tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "scheduling",
    srcs = ["scheduling.cc"],
    hdrs = ["scheduling.h"],
    deps = [
        ":tile",
        ":tiled_hlo",
        "//xla:util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "scheduling_test",
    srcs = ["scheduling_test.cc"],
    deps = [
        ":scheduling",
        ":tile",
        ":tiled_hlo",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tiled_hlo",
    srcs = ["tiled_hlo.cc"],
    hdrs = ["tiled_hlo.h"],
    deps = [
        ":tile",
        ":tile_propagation",
        "//xla:util",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/service:name_uniquer",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "tiled_hlo_test",
    srcs = ["tiled_hlo_test.cc"],
    deps = [
        ":test_utils",
        ":tile",
        ":tile_propagation",
        ":tiled_hlo",
        "//xla:util",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

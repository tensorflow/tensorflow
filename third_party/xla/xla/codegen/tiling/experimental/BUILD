load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

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
    name = "symbolic_tile",
    srcs = [
        "symbolic_tile.cc",
        "tiling_space.cc",
    ],
    hdrs = [
        "symbolic_tile.h",
        "tiling_space.h",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/tiling:constraint_expression",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "symbolic_tile_propagation",
    srcs = ["symbolic_tile_propagation.cc"],
    hdrs = ["symbolic_tile_propagation.h"],
    deps = [
        ":symbolic_tile",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:matmul_indexing_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "symbolic_tile_propagation_test",
    srcs = ["symbolic_tile_propagation_test.cc"],
    deps = [
        ":symbolic_tile",
        ":symbolic_tile_propagation",
        ":test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/log:check",
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
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_test_utils",
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
    name = "symbolic_tiled_hlo",
    srcs = ["symbolic_tiled_hlo.cc"],
    hdrs = ["symbolic_tiled_hlo.h"],
    deps = [
        ":symbolic_tile",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tiled_hlo_test",
    srcs = ["symbolic_tiled_hlo_test.cc"],
    deps = [
        ":symbolic_tile",
        ":symbolic_tile_propagation",
        ":symbolic_tiled_hlo",
        ":test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
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
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "symbolic_tiled_hlo_computation",
    srcs = ["symbolic_tiled_hlo_computation.cc"],
    hdrs = ["symbolic_tiled_hlo_computation.h"],
    deps = [
        ":symbolic_tile",
        ":symbolic_tile_propagation",
        ":symbolic_tiled_hlo",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:instruction_fusion",
        "//xla/service:name_uniquer",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "symbolic_tiled_hlo_computation_test",
    srcs = ["symbolic_tiled_hlo_computation_test.cc"],
    deps = [
        ":symbolic_tiled_hlo_computation",
        ":test_utils",
        "//xla:util",
        "//xla/codegen/tiling/experimental:symbolic_tiled_hlo",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

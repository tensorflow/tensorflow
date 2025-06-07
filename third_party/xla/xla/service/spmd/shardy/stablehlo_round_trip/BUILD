# Import/Export passes for going from `sdy.sharding`s to `mhlo.sharding`s and vice versa.

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
    name = "export_shardings",
    srcs = ["export_shardings.cc"],
    hdrs = ["export_shardings.h"],
    deps = [
        "//xla:array",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "export_ops",
    srcs = ["export_ops.cc"],
    hdrs = ["export_ops.h"],
    deps = [
        "//xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "shard_map_export",
    srcs = ["shard_map_export.cc"],
    hdrs = ["shard_map_export.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "export_callback_custom_calls",
    srcs = ["export_callback_custom_calls.cc"],
    hdrs = ["export_callback_custom_calls.h"],
    deps = [
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "stablehlo_export",
    srcs = ["stablehlo_export.cc"],
    hdrs = ["stablehlo_export.h"],
    deps = [
        ":export_callback_custom_calls",
        ":export_manual_reduction_collectives_cc",
        ":export_ops",
        ":export_shardings",
        ":shard_map_export",
        "//xla/service/spmd/shardy/round_trip_common:export_named_computations",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "stablehlo_import",
    srcs = ["stablehlo_import.cc"],
    hdrs = ["stablehlo_import.h"],
    deps = [
        ":shard_map_import",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/translate/mhlo_to_hlo:attribute_exporter",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy/round_trip_common:pipeline_passes",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

xla_cc_test(
    name = "stablehlo_import_test",
    srcs = ["stablehlo_import_test.cc"],
    deps = [
        ":stablehlo_import",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/ir:register",
    ],
)

cc_library(
    name = "shard_map_import",
    srcs = ["shard_map_import.cc"],
    hdrs = ["shard_map_import.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "export_manual_reduction_collectives_cc",
    srcs = ["export_manual_reduction_collectives.cc"],
    hdrs = ["export_manual_reduction_collectives.h"],
    deps = [
        "//xla:array",
        "//xla/service/spmd/shardy:utils",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

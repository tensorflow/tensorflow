# Import/Export passes for going from `sdy.sharding`s to `mhlo.sharding`s and vice versa.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/deepmind/partir/...",
        "//third_party/openxla/shardy/tools/...",
        "//xla/service/spmd/shardy/...",
    ],
)

cc_library(
    name = "export_shardings",
    srcs = ["export_shardings.cc"],
    hdrs = ["export_shardings.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service/spmd/shardy:constants",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/algorithm:container",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

cc_library(
    name = "export_ops",
    srcs = ["export_ops.cc"],
    hdrs = ["export_ops.h"],
    deps = [
        "//xla:sharding_op_util",
        "//xla/mlir_hlo",
        "//xla/service/spmd/shardy:constants",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

cc_library(
    name = "shard_map_export",
    srcs = ["shard_map_export.cc"],
    hdrs = ["shard_map_export.h"],
    deps = [
        ":export_shardings",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

cc_library(
    name = "mhlo_export",
    srcs = ["mhlo_export.cc"],
    hdrs = ["mhlo_export.h"],
    deps = [
        ":export_ops",
        ":export_shardings",
        ":shard_map_export",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mhlo_import",
    srcs = ["mhlo_import.cc"],
    hdrs = ["mhlo_import.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy/round_trip_common:pipeline_passes",
        "//xla/translate/mhlo_to_hlo:attribute_exporter",
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

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/deepmind/partir/compiler/mpmd/...",
        "//learning/deepmind/partir/compiler/shardonnay/...",
        "//third_party/openxla/shardy/tools/...",
        "//third_party/py/jax/...",
        "//xla/...",
    ],
)

cc_library(
    name = "export_shardy_attrs",
    srcs = ["export_shardy_attrs.cc"],
    hdrs = ["export_shardy_attrs.h"],
    deps = [
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
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
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
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
    name = "import_shardy_attrs",
    srcs = ["import_shardy_attrs.cc"],
    hdrs = ["import_shardy_attrs.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
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
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
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
    name = "shard_map_import",
    srcs = ["shard_map_import.cc"],
    hdrs = ["shard_map_import.h"],
    deps = [
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
    name = "remove_size_one_axes",
    srcs = ["remove_size_one_axes.cc"],
    hdrs = ["remove_size_one_axes.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/common:sharding_walker",
    ],
)

cc_library(
    name = "import_callback_custom_calls",
    srcs = ["import_callback_custom_calls.cc"],
    hdrs = ["import_callback_custom_calls.h"],
    deps = [
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "dedup_meshes",
    srcs = ["dedup_meshes.cc"],
    hdrs = ["dedup_meshes.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/common:sharding_walker",
    ],
)

cc_library(
    name = "pipelines",
    srcs = ["pipelines.cc"],
    hdrs = ["pipelines.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":dedup_meshes",
        ":export_ops",
        ":export_shardy_attrs",
        ":import_callback_custom_calls",
        ":import_shardy_attrs",
        ":remove_size_one_axes",
        ":shard_map_export",
        ":shard_map_import",
        "//xla/service:hlo_proto_cc",
        "//xla/service/spmd/shardy/round_trip_common:export_named_computations",
        "//xla/service/spmd/shardy/round_trip_common:pipeline_passes",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

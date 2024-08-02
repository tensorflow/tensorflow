load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/deepmind/partir/compiler/shardonnay/...",
        "//third_party/openxla/shardy/tools/...",
        "//xla/...",
    ],
)

cc_library(
    name = "export_shardings",
    srcs = ["export_shardings.cc"],
    hdrs = ["export_shardings.h"],
    deps = [
        "//xla/mlir_hlo",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
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
    name = "import_shardings",
    srcs = ["import_shardings.cc"],
    hdrs = ["import_shardings.h"],
    deps = [
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

cc_library(
    name = "pipelines",
    srcs = ["pipelines.cc"],
    hdrs = ["pipelines.h"],
    deps = [
        ":export_ops",
        ":export_shardings",
        ":import_shardings",
        "//xla/service:hlo_proto_cc",
        "//xla/service/spmd/shardy/mhlo_round_trip:export_shardings",
        "//xla/service/spmd/shardy/round_trip_common:pipeline_passes",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//xla/service/spmd/shardy/...",
    ],
)

cc_library(
    name = "convert_sharding_custom_calls",
    srcs = ["convert_sharding_custom_calls.cc"],
    hdrs = ["convert_sharding_custom_calls.h"],
    deps = [
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/ir:dialect",
        "//xla:sharding_op_util",
        "//xla/mlir_hlo",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "identity_to_pass_through_while_args",
    srcs = ["identity_to_pass_through_while_args.cc"],
    hdrs = ["identity_to_pass_through_while_args.h"],
    deps = [
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/ir:dialect",
        "//xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "import_constants",
    srcs = ["import_constants.cc"],
    hdrs = ["import_constants.h"],
    deps = [
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/ir:dialect",
        "//xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "shard_map_import",
    srcs = ["shard_map_import.cc"],
    hdrs = ["shard_map_import.h"],
    deps = [
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/ir:dialect",
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/algorithm:container",
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
    ],
)

cc_library(
    name = "pipeline_passes",
    srcs = ["pipeline_passes.cc"],
    hdrs = ["pipeline_passes.h"],
    deps = [
        ":convert_sharding_custom_calls",
        ":identity_to_pass_through_while_args",
        ":import_constants",
        ":shard_map_import",
        "//xla/mlir_hlo:mhlo_passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

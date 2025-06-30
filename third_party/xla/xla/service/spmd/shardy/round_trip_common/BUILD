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
    name = "import_sdy_custom_calls",
    srcs = ["import_sdy_custom_calls.cc"],
    hdrs = ["import_sdy_custom_calls.h"],
    deps = [
        "//xla:sharding_op_util",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
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
    name = "export_named_computations",
    srcs = ["export_named_computations.cc"],
    hdrs = ["export_named_computations.h"],
    deps = [
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
    name = "import_uninlineable_func_calls",
    srcs = ["import_uninlineable_func_calls.cc"],
    hdrs = ["import_uninlineable_func_calls.h"],
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
    ],
)

cc_library(
    name = "import_constants",
    srcs = ["import_constants.cc"],
    hdrs = ["import_constants.h"],
    deps = [
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
    name = "open_while_free_vars_sharding",
    srcs = ["open_while_free_vars_sharding.cc"],
    hdrs = ["open_while_free_vars_sharding.h"],
    deps = [
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
    name = "pipeline_passes",
    srcs = ["pipeline_passes.cc"],
    hdrs = ["pipeline_passes.h"],
    deps = [
        ":import_constants",
        ":import_sdy_custom_calls",
        ":import_uninlineable_func_calls",
        ":open_while_free_vars_sharding",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

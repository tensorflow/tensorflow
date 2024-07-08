# Description: Integrates Shardy into the current XLA pipeline.

load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_binary(
    name = "xla_sdy_opt",
    srcs = ["sdy_opt_main.cc"],
    deps = [
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/ir:dialect",
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/transforms:passes",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:stablehlo_ops",
    ],
)

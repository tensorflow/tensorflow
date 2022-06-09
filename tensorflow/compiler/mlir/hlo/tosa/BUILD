# MHLO -> TOSA bridge.

package(
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [],
)

cc_binary(
    name = "mhlo-tosa-opt",
    srcs = ["mhlo_tosa_opt.cc"],
    deps = [
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
    ],
)

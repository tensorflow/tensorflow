load("//xla:lit.bzl", "lit_test_suite")
load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "mlir_fusions_opt",
    srcs = ["mlir_fusions_opt.cc"],
    deps = [
        "//xla/mlir_hlo",
        "//xla/service/gpu/fusions/mlir:passes",
        "//xla/service/gpu/fusions/mlir/ir:xla_gpu",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
)

lit_test_suite(
    name = "tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":mlir_fusions_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

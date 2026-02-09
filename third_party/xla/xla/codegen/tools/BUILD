load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "emitters_opt",
    srcs = ["emitters_opt.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    visibility = [
        "//xla/backends/cpu/codegen:__subpackages__",
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/codegen:__subpackages__",
        "//xla/codegen/emitters/ir/tests:__subpackages__",
        "//xla/codegen/emitters/tests:__pkg__",
        "//xla/codegen/emitters/transforms/tests:__subpackages__",
        "//xla/codegen/xtile/ir/tests:__pkg__",
    ],
    deps = [
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/backends/cpu/codegen/emitters/transforms:passes",
        "//xla/backends/cpu/codegen/tiled/transforms:passes",
        "//xla/backends/gpu/codegen/emitters:emitter_base",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/backends/gpu/codegen/emitters/transforms:passes",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/transforms:lower_to_llvm_gpu_pass",
        "//xla/codegen/emitters/transforms:pass_pipelines",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/codegen/xtile/ir/transforms:passes",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/mlir_hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "test_lib",
    testonly = True,
    srcs = ["test_lib.cc"],
    hdrs = ["test_lib.h"],
    visibility = [
        "//xla/backends/cpu/codegen/tools:__subpackages__",
        "//xla/backends/gpu/codegen/tools:__subpackages__",
    ],
    deps = [
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/tools:hlo_module_loader",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

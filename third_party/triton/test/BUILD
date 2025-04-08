# copybara:uncomment_begin
# load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")
# load("//tools/build_defs/build_test:build_test.bzl", "build_test")
#
# package(
#     default_applicable_licenses = ["@triton//:license"],
#     default_compatible_with = ["//buildenv/target:non_prod"],
#     default_visibility = ["@triton//:__subpackages__"],
# )
#
# glob_lit_tests(
#     name = "all_tests",
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//llvm:llc",
#         "@llvm-project//llvm:opt",
#         "@llvm-project//mlir:mlir-translate",
#         "@triton//:triton-llvm-opt",
#         "@triton//:triton-opt",
#         "@triton//:triton-tensor-layout",
#     ],
#     driver = "@llvm-project//mlir:run_lit.sh",
#     exclude = [
#         # broken, offending change reverted in
#         # https://github.com/triton-lang/triton/commit/3ed479f2f91a1d94dacb547115d357f5ce3219d8
#         "Conversion/reduce_to_llvm.mlir",
#         "Conversion/amd/dedup-by-constancy.mlir",  # AMD-specific, broken
#         "TritonGPU/amd/amd-instruction-sched.mlir",  # AMD-specific, broken with -debug-only.
#         "TritonGPU/optimize_epilogue.mlir",  # TODO: b/346283526 - AMD-specific, triggering UBSAN
#         # Broken between https://github.com/triton-lang/triton/commit/0dc2154e34ad0eb8d60ff2534755954aa8c8f20e
#         # and https://github.com/triton-lang/triton/commit/196a08f04b92fcf0e52015d3b1068c18e4eea5b5
#         "TritonGPU/loop-pipeline.mlir",
#     ],
#     test_file_exts = [
#         "mlir",
#         "ll",
#     ],
# )
#
# build_test(
#     name = "build_test",
#     allow_empty_target = False,
#     targets = [
#         "@triton//:TritonDialects",
#         "@triton//:TritonGPUToLLVM",
#         "@triton//:TritonGPUTransforms",
#         "@triton//:TritonLLVMIR",
#         "@triton//:TritonPTX",
#         "@triton//:TritonToTritonGPU",
#         "@triton//:TritonTools",
#         "@triton//:TritonTransforms",
#         "@triton//:triton-opt",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "TritonTestAnalysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@triton//:TritonDialects",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

exports_files(srcs = [
    "lib/Dialect/TritonGPU/TestTC05MMAPipeline.cpp",
])

#ifndef TRITONAMDGPU_CONVERSION_PASSES
#define TRITONAMDGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def OptimizeAMDLDSUsage : Pass<"optimize-amd-lds-usage", "mlir::ModuleOp"> {
    let summary = "Minimize LDS usage";
    let constructor = "mlir::triton::AMD::createOptimizeLDSUsagePass(\"\")";

    let options = [
        Option<"targetArch", "target-arch", "std::string", /*default*/"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"customLDSLimit", "lds-limit", "int", /*default*/"0",
               "custom limit of LDS consumption, if not provided, maximum LDS size is used">,
    ];
}

def ConvertTritonAMDGPUToLLVM : Pass<"convert-triton-amdgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonAMDGPUToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def ConvertBuiltinFuncToLLVM : Pass<"convert-builtin-func-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Builtin Func to LLVM";
    let constructor = "mlir::triton::createConvertBuiltinFuncToLLVMPass(/*ftz=*/true)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];

    let options = [
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def TritonAMDGPUInsertInstructionSchedHints : Pass<"triton-amdgpu-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Insert instruction scheduling hints after the dot ops in the main loop";
    let constructor = "mlir::triton::createTritonAMDGPUInsertInstructionSchedHintsPass(/*variant=*/\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::amdgpu::TritonAMDGPUDialect"];

    let options = [
        Option<"variant", "variant", "std::string", /*default*/"\"none\"",
               "instruction scheduling variant">,
    ];
}

def TritonAMDGPULowerInstructionSchedHints : Pass<"triton-amdgpu-lower-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Lower instruction scheduling hints to LLVM intrinsics";
    let constructor = "mlir::triton::createTritonAMDGPULowerInstructionSchedHintsPass(/*arch=*/\"\",/*numStages=*/2)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::amdgpu::TritonAMDGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"numStages", "num_stages", "int32_t", /*default*/"2",
                "number of pipeline stages">,
    ];
}

#endif

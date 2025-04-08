#ifndef TRITON_TARGET_LLVMIR_PASSES
#define TRITON_TARGET_LLVMIR_PASSES

include "mlir/Pass/PassBase.td"

def LLVMDIScope: Pass<"enable-line-info", "mlir::ModuleOp"> {
  let summary = "Materialize LLVM line info";
  let description = [{
    This pass materializes line mapping information for LLVM IR dialect operations.
  }];

  let constructor = "mlir::createLLVMDIScopePass()";
}

#endif

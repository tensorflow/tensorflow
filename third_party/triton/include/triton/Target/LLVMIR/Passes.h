#ifndef TRITON_TARGET_LLVM_IR_PASSES_H
#define TRITON_TARGET_LLVM_IR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Create a pass to add DIScope
std::unique_ptr<Pass> createLLVMDIScopePass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "triton/Target/LLVMIR/Passes.h.inc"

} // namespace mlir

#endif // TRITON_TARGET_LLVM_IR_PASSES_H

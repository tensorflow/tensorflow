#ifndef TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton::gpu {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

} // namespace triton::gpu

} // namespace mlir

#endif

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

} // namespace gpu
} // namespace triton
} // namespace mlir
#endif

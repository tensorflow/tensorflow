#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createCombineOpsPass();

std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();
std::unique_ptr<Pass> createReorderBroadcastPass();
std::unique_ptr<Pass> createRewriteTensorPointerPass();
std::unique_ptr<Pass> createLoopUnrollPass();

} // namespace triton

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace mlir

#endif

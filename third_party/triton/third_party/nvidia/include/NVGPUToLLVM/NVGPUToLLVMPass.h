#ifndef TRITON_CONVERSION_NVGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_NVGPU_TO_LLVM_PASS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

namespace nvgpu {

using Constraints = std::vector<std::string>;
using OperandsAndConstraints = std::vector<std::pair<Value, std::string>>;

LogicalResult
rewriteAsPtxAsm(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                std::string ptxAsm,
                const OperandsAndConstraints &operandsAndConstraints = {},
                const Constraints &outputConstraints = {});

} // namespace nvgpu

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUToLLVMPass();

} // namespace triton

} // namespace mlir

#endif

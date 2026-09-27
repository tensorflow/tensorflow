/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/WalkResult.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTVERIFYSHARDINGSPECIFIEDPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

bool IsArrayWithUnspecifiedSharding(mlir::Type type) {
  auto array_type = llvm::dyn_cast_or_null<xla::ifrt::IfrtArrayType>(type);
  if (array_type == nullptr) {
    return false;
  }
  return IsUnspecifiedSharding(array_type.getShardingAttr());
}

class IfrtVerifyShardingSpecifiedPass
    : public impl::IfrtVerifyShardingSpecifiedPassBase<
          IfrtVerifyShardingSpecifiedPass> {
  using IfrtVerifyShardingSpecifiedPassBase::
      IfrtVerifyShardingSpecifiedPassBase;

 public:
  void runOnOperation() override;
};

void IfrtVerifyShardingSpecifiedPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // Only IFRT functions have IFRT types.
  if (!IsIfrtFunction(func_op)) {
    return;
  }
  mlir::FunctionType func_type = func_op.getFunctionType();
  for (const auto [idx, input_type] : llvm::enumerate(func_type.getInputs())) {
    if (IsArrayWithUnspecifiedSharding(input_type)) {
      func_op->emitOpError()
          << "argument " << idx << " has unspecified sharding.";
      return signalPassFailure();
    }
  }
  for (const auto [idx, result_type] :
       llvm::enumerate(func_type.getResults())) {
    if (IsArrayWithUnspecifiedSharding(result_type)) {
      func_op->emitOpError()
          << "result " << idx << " has unspecified sharding.";
      return signalPassFailure();
    }
  }

  mlir::WalkResult result = func_op.walk([&](mlir::Operation* op) {
    for (const auto [idx, operand_type] :
         llvm::enumerate(op->getOperandTypes())) {
      if (IsArrayWithUnspecifiedSharding(operand_type)) {
        op->emitOpError() << "argument " << idx << " has unspecified sharding.";
        return mlir::WalkResult::interrupt();
      }
    }
    for (const auto [idx, result_type] :
         llvm::enumerate(op->getResultTypes())) {
      if (IsArrayWithUnspecifiedSharding(result_type)) {
        op->emitOpError() << "result " << idx << " has unspecified sharding.";
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return signalPassFailure();
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

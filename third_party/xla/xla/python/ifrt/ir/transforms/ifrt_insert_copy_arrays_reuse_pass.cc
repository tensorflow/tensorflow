/* Copyright 2026 The OpenXLA Authors.

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

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTINSERTCOPYARRAYSREUSEPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

class IfrtInsertCopyArraysReusePass
    : public impl::IfrtInsertCopyArraysReusePassBase<
          IfrtInsertCopyArraysReusePass> {
 public:
  void runOnOperation() override;
};

void IfrtInsertCopyArraysReusePass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!IsIfrtFunction(func_op)) {
    return;
  }

  func_op.walk([&](mlir::func::ReturnOp return_op) {
    mlir::OpBuilder builder(return_op);
    llvm::DenseMap<mlir::Value, int> value_counts;

    for (int i = 0; i < return_op.getNumOperands(); ++i) {
      mlir::Value val = return_op.getOperand(i);
      if (llvm::isa<IfrtArrayType>(val.getType())) {
        int count = value_counts[val]++;
        if (count > 0) {
          CopyArraysOp copy_arrays_op = CopyArraysOp::create(
              builder, return_op.getLoc(),
              /*outputs=*/{val.getType()},
              /*control_output=*/IfrtControlType::get(builder.getContext()),
              /*inputs=*/{val},
              /*donated=*/false,
              /*reuse=*/true,
              /*control_inputs=*/{});
          return_op.setOperand(i, copy_arrays_op.getOutputs().front());
        }
      }
    }
  });
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

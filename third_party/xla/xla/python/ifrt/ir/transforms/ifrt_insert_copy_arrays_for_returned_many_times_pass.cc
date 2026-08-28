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
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTINSERTCOPYARRAYSFORRETURNEDMANYTIMESPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

using mlir::func::FuncOp;

class IfrtInsertCopyArraysForReturnedManyTimesPass
    : public impl::IfrtInsertCopyArraysForReturnedManyTimesPassBase<
          IfrtInsertCopyArraysForReturnedManyTimesPass> {
 public:
  void runOnOperation() override;
};

void IfrtInsertCopyArraysForReturnedManyTimesPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!IsIfrtFunction(func_op)) {
    return;
  }

  // Copy the arrays that are returned multiple times from the program. This is
  // necessary to ensure that: 1) donated arrays are deleted, but output arrays
  // remain valid, and 2) calling `Delete()` on input arrays does not delete the
  // returned arrays.
  func_op.walk([&](mlir::func::ReturnOp return_op) {
    mlir::OpBuilder builder(return_op);

    // Mapping from value to the number of times it is returned.
    llvm::DenseMap<mlir::Value, int> val_to_num_returns;
    for (mlir::Value val : return_op.getOperands()) {
      if (llvm::isa<IfrtArrayType>(val.getType())) {
        val_to_num_returns[val]++;
      }
    }

    for (int i = 0; i < return_op.getNumOperands(); ++i) {
      mlir::Value val = return_op.getOperand(i);
      if (!llvm::isa<IfrtArrayType>(val.getType())) {
        continue;
      }

      mlir::BlockArgument block_arg = mlir::dyn_cast<mlir::BlockArgument>(val);
      bool is_block_arg = block_arg && block_arg.getOwner() == &func_op.front();

      bool is_last_return = --val_to_num_returns[val] == 0;
      if (!is_block_arg && is_last_return) {
        // Do not insert a copy for the last return of a non-block argument.
        continue;
      }

      bool is_donatable =
          is_last_return &&
          (!is_block_arg ||
           func_op.getArgAttr(block_arg.getArgNumber(),
                              kIfrtDonatedArgAttrName) != nullptr);
      CopyArraysOp copy_op = CopyArraysOp::create(
          builder, return_op.getLoc(),
          /*outputs=*/{val.getType()},
          /*control_output=*/IfrtControlType::get(builder.getContext()),
          /*inputs=*/{val},
          /*donated=*/is_donatable,
          /*reuse=*/false,
          /*control_inputs=*/{});
      return_op.setOperand(i, copy_op.getOutputs().front());
    }
  });
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

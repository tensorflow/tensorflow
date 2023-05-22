/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_REPLICATETENSORLISTINITOPSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Replicates the TensorList initialization ops for all the uses.
// No need to delete the original TensorList as it might be used elsewhere.
template <typename T>
void ReplicateTensorListForUses(T tensor_list_op) {
  Value tensor_list = tensor_list_op.getResult();
  std::vector<OpOperand*> uses;
  for (auto& use : tensor_list.getUses()) {
    uses.emplace_back(&use);
  }
  OpBuilder builder(tensor_list_op.getOperation());
  for (OpOperand* operand : uses) {
    auto new_op = builder.clone(*tensor_list_op.getOperation());
    operand->set(new_op->getResult(0));
  }
}

// This transformation pass replicates TensorList initialization ops.
class ReplicateTensorListInitOps
    : public impl::ReplicateTensorListInitOpsPassBase<
          ReplicateTensorListInitOps> {
 public:
  void runOnOperation() override {
    getOperation().walk([](Operation* op) {
      if (auto tl_reserve = dyn_cast<TensorListReserveOp>(op)) {
        ReplicateTensorListForUses(tl_reserve);
      }
      if (auto tl_empty = dyn_cast<EmptyTensorListOp>(op)) {
        ReplicateTensorListForUses(tl_empty);
      }
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicateTensorListInitOpsPass() {
  return std::make_unique<ReplicateTensorListInitOps>();
}

}  // namespace TF
}  // namespace mlir

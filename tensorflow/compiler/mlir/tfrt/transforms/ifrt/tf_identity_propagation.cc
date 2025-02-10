/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFIDENTITYPROPAGATIONPASS
#define GEN_PASS_DECL_TFIDENTITYPROPAGATIONPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

constexpr absl::string_view kXlaShardingAttr = "_XlaSharding";

bool IsTerminator(mlir::Operation* op) {
  return op->hasTrait<mlir::OpTrait::IsTerminator>();
}

class TfIdentityPropagationPass
    : public impl::TfIdentityPropagationPassBase<TfIdentityPropagationPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    func.walk([](mlir::TF::IdentityOp identity) {
      // Don't propagate inputs of identity ops with sharding annotation since
      // identity ops are sometimes used to change output sharding.
      if (identity->hasAttr(kXlaShardingAttr)) {
        return;
      }
      // Identity outputs to terminator ops (e.g., `func.return`) cannot be
      // replaced unless input/output types are exactly the same. Doing so may
      // cause mismatch between the enclosing region's return type and the
      // terminator's arg type.
      const bool same_type =
          identity.getInput().getType() == identity.getOutput().getType();
      identity.getOutput().replaceUsesWithIf(
          identity.getInput(), [&](mlir::OpOperand& operand) {
            return same_type || !IsTerminator(operand.getOwner());
          });
    });

    func.walk([](mlir::TF::IdentityNOp identity_n) {
      if (identity_n->hasAttr(kXlaShardingAttr)) {
        return;
      }
      for (auto [input, output] :
           llvm::zip(identity_n.getInput(), identity_n.getOutput())) {
        const bool same_type = input.getType() == output.getType();
        output.replaceUsesWithIf(input, [&](mlir::OpOperand& operand) {
          return same_type || !IsTerminator(operand.getOwner());
        });
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfIdentityPropagationPass() {
  return std::make_unique<TfIdentityPropagationPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/legacy_call/pass.h"

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {
namespace {
class LiftLegacyCallPass : public LiftLegacyCallBase<LiftLegacyCallPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    tfg_legacy_call_id_ = StringAttr::get(context, "tfg.legacy_call");
    return success();
  }

  void runOnOperation() override {
    FunctionTable table(getOperation());
    for (Operation &op : getOperation().getOps()) {
      op.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::IntrinsicOperation>() ||
            !table.IsLegacyCall(op))
          return;

        op->setAttr(tfg_legacy_call_id_,
                    FlatSymbolRefAttr::get(&getContext(),
                                           op->getName().stripDialect()));
      });
    }
  }

 private:
  // The cached identifier of the legacy call tag.
  StringAttr tfg_legacy_call_id_;
};
}  // namespace
std::unique_ptr<Pass> CreateLiftLegacyCallPass() {
  return std::make_unique<LiftLegacyCallPass>();
}
}  // namespace tfg
}  // namespace mlir

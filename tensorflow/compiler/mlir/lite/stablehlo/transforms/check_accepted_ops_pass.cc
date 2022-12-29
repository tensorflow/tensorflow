/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/check_accepted_ops_pass.h"

#include <algorithm>
#include <string>
#include <vector>

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_util.h"

namespace mlir {
namespace odml {

// Define a pass class for checking if non-accepted dialect ops exist or not.
namespace {
class CheckAcceptedOpsPass
    : public PassWrapper<CheckAcceptedOpsPass, OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckAcceptedOpsPass)

  explicit CheckAcceptedOpsPass(
      const std::vector<std::string> &optional_accepted_dialects)
      : accepted_dialects_(GetAcceptedDialects()),
        optional_accepted_dialects_(optional_accepted_dialects) {}

  // Check if TF dialect ops exist over the module.
  void runOnOperation() override;

 private:
  std::vector<std::string> accepted_dialects_;
  std::vector<std::string> optional_accepted_dialects_;
};
}  // namespace

void CheckAcceptedOpsPass::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    auto dialect_name = op->getDialect()->getNamespace();
    auto op_name = op->getName().stripDialect();
    if (IsAcceptedOp(dialect_name, op_name, accepted_dialects_)) {
      // If given op is in the `accepted_dialects_`, it's ok.
    } else if (IsAcceptedOp(dialect_name, op_name,
                            optional_accepted_dialects_)) {
      // If the given op is in the `optional_accepted_dialects_`, let's warn it.
      op->emitWarning() << op->getName().getStringRef() << " op is temporarily "
                        << "accepted, but it should be removed in the end.";
    } else {
      // The other ops are not accepted.
      return signalPassFailure();
    }
  });
}

}  // namespace odml
}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::odml::createCheckAcceptedOpsPass(
    const std::vector<std::string> &optional_accepted_dialects) {
  return std::make_unique<mlir::odml::CheckAcceptedOpsPass>(
      optional_accepted_dialects);
}

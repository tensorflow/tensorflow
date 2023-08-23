/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::tosa {

#define GEN_PASS_DEF_VERIFYFULLYCONVERTED
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

static void emitLegalizationErrors(Location loc,
                                   const DenseSet<Operation *> &illegalOps) {
  // Print op errors for each of the illegal ops that still remain.
  llvm::MapVector<StringRef, int> opNameCounts;
  for (Operation *illegalOp : illegalOps) {
    StringRef opName = illegalOp->getName().getStringRef();
    opNameCounts[opName]++;
    illegalOp->emitOpError() << ": illegal op still exists";
  }

  std::vector<std::string> errorMessages;
  errorMessages.reserve(opNameCounts.size());
  for (const auto &opInfo : opNameCounts) {
    errorMessages.push_back(
        llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
  }
  emitError(loc) << "The following illegal operations still remain: \n"
                 << llvm::join(errorMessages, "\n") << "\n";
}

LogicalResult verifyAllOperationsAreLegal(Operation *op,
                                          const ConversionTarget &target) {
  DenseSet<Operation *> illegalOps;
  op->walk([&](Operation *op) {
    if (!target.isLegal(op)) {
      illegalOps.insert(op);
    }
  });
  if (illegalOps.empty()) return success();
  emitLegalizationErrors(op->getLoc(), illegalOps);
  return failure();
}

class VerifyFullyConvertedPass
    : public impl::VerifyFullyConvertedBase<VerifyFullyConvertedPass> {
 public:
  // Validates that no TFLite frontends ops are in the function.
  void runOnOperation() override {
    // We don't just use applyPartialConversion with no patterns because this
    // pass shouldn't alter the IR at all (including via folding or
    // canonicalizations that dialect conversion does automatically).
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalDialect<mlir::TFL::TensorFlowLiteDialect>();
    target.addIllegalOp<mlir::UnrealizedConversionCastOp>();
    if (failed(verifyAllOperationsAreLegal(getOperation(), target)))
      return signalPassFailure();
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace mlir::tosa

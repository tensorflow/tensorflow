/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for legalizing HLO to TensorFlow.

#include <memory>

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace TF {
namespace {

class LegalizeHloToTf : public FunctionPass<LegalizeHloToTf> {
 public:
  LegalizeHloToTf() = default;
  LegalizeHloToTf(const LegalizeHloToTf &) {}

  /// Performs the legalization to the TF dialect.
  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_legalize_hlo.inc"

/// Performs the lowering to XLA dialect.
void LegalizeHloToTf::runOnFunction() {
  MLIRContext &context = getContext();

  // Add legalization patterns to the list.
  OwningRewritePatternList patterns;
  populateWithGenerated(&context, &patterns);

  ConversionTarget target(context);
  target.addLegalDialect<TensorFlowDialect>();
  target.addLegalOp<CallOp>();
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

static PassRegistration<LegalizeHloToTf> pass(
    "tf-legalize-hlo", "Legalize from HLO to the TF dialect");

}  // end namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateLegalizeHloToTfPass() {
  return std::make_unique<LegalizeHloToTf>();
}

}  // end namespace TF
}  // end namespace mlir

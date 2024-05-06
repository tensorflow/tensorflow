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
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_avg_pool.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

namespace {

// This file is generated from `passes.td` and provides the implementation base
// class.
#define GEN_PASS_DEF_COMPOSITELOWERINGPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

class CompositeLoweringPass
    : public impl::CompositeLoweringPassBase<CompositeLoweringPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompositeLoweringPass);

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_composite_lowering.inc"

void CompositeLoweringPass::runOnOperation() {
  MLIRContext& context = getContext();
  RewritePatternSet patterns(&getContext());

  populateWithGenerated(patterns);

  ConversionTarget target(context);
  target.addLegalDialect<TFL::TensorFlowLiteDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("Composite lowering pass failed.");
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateCompositeLoweringPass() {
  return std::make_unique<CompositeLoweringPass>();
}

// Registers the pass implementation
static PassRegistration<CompositeLoweringPass> pass;

}  // namespace odml
}  // namespace mlir

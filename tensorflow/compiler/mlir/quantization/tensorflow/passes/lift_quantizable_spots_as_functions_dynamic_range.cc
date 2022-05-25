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
#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class LiftQuantizableSpotsAsFunctionsDynamicRangePass
    : public PassWrapper<LiftQuantizableSpotsAsFunctionsDynamicRangePass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LiftQuantizableSpotsAsFunctionsDynamicRangePass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-lift-quantizable-spots-as-functions-dynamic-range";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Replace quantization candidates with composite functions into the "
           "module for post-training dynamic range case";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;
};

static PassRegistration<LiftQuantizableSpotsAsFunctionsDynamicRangePass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/lift_quantizable_spots_as_functions_dynamic_range.inc"

void LiftQuantizableSpotsAsFunctionsDynamicRangePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  populateWithGenerated(patterns);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  for (auto func : module.getOps<func::FuncOp>()) {
    if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
      func.emitError()
          << "quant-lift-quantizable-spots-as-functions-dynamic-range failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsDynamicRangePass() {
  return std::make_unique<LiftQuantizableSpotsAsFunctionsDynamicRangePass>();
}

}  // namespace quant
}  // namespace mlir

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
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class LiftQuantizableSpotsAsFunctionsDRQPass
    : public PassWrapper<LiftQuantizableSpotsAsFunctionsDRQPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LiftQuantizableSpotsAsFunctionsDRQPass)

  // Constructor used by the PassRegistration. This is only used by test.
  explicit LiftQuantizableSpotsAsFunctionsDRQPass() = default;

  // Constructor used by manually creating the pass.
  explicit LiftQuantizableSpotsAsFunctionsDRQPass(
      int min_num_elements_for_weights) {
    min_num_elements_for_weights_ = min_num_elements_for_weights;
  }

  LiftQuantizableSpotsAsFunctionsDRQPass(
      const LiftQuantizableSpotsAsFunctionsDRQPass& other) {
    min_num_elements_for_weights_ = other.min_num_elements_for_weights_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-lift-quantizable-spots-as-functions-drq";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Replace quantization candidates with composite functions into the "
           "module for post-training dynamic range case";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;

  Option<int64_t> min_num_elements_for_weights_{
      *this, "min-num-elements-for-weights", llvm::cl::init(0),
      llvm::cl::desc("The minimum required number of elements in a weight "
                     "array to apply quantization.")};
};

class CheckQuantizableOps
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit CheckQuantizableOps(MLIRContext* context,
                               int min_num_elements_for_weights)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        min_num_elements_for_weights_(min_num_elements_for_weights) {}

 private:
  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(call_op);
    if (spec->quantizable_operands.empty()) return failure();

    for (auto idx : spec->quantizable_operands) {
      // This op is guaranteed to be a constant as ODS checks IsConstTensor.
      // Check if the number of elements meets the requirement.
      int current_num_elements =
          call_op.getOperand(idx).getType().cast<ShapedType>().getNumElements();
      if (current_num_elements < min_num_elements_for_weights_) {
        call_op.emitRemark("Quantization is skipped for ")
            << call_op->getName().getStringRef().str() << " because it has "
            << current_num_elements
            << " elements which is fewer than the threshold("
            << min_num_elements_for_weights_ << " elements).";
        call_op->removeAttr(kQuantTraitAttrName);
      }
    }
    return failure();
  }

  int min_num_elements_for_weights_;
};

static PassRegistration<LiftQuantizableSpotsAsFunctionsDRQPass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/lift_quantizable_spots_as_functions_drq.inc"

void LiftQuantizableSpotsAsFunctionsDRQPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  populateWithGenerated(patterns);
  patterns.add<CheckQuantizableOps>(ctx, min_num_elements_for_weights_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  for (auto func : module.getOps<func::FuncOp>()) {
    if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
      func.emitError()
          << "quant-lift-quantizable-spots-as-functions-drq failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsDRQPass(int min_num_elements_for_weights) {
  return std::make_unique<LiftQuantizableSpotsAsFunctionsDRQPass>(
      min_num_elements_for_weights);
}

}  // namespace quant
}  // namespace mlir

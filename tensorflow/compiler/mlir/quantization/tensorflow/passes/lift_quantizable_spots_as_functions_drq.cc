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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

using QuantMethod =
    ::tensorflow::quantization::QuantizationMethod::PresetMethod;
using ::tensorflow::quantization::OpSet;

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
      const QuantMethod quantization_method, const OpSet target_opset,
      const int min_num_elements_for_weights) {
    quantization_method_ = quantization_method;
    target_opset_ = target_opset;
    min_num_elements_for_weights_ = min_num_elements_for_weights;
  }

  LiftQuantizableSpotsAsFunctionsDRQPass(
      const LiftQuantizableSpotsAsFunctionsDRQPass& other) {
    quantization_method_ = other.quantization_method_;
    target_opset_ = other.target_opset_;
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

 private:
  Option<OpSet> target_opset_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};

  Option<int64_t> min_num_elements_for_weights_{
      *this, "min-num-elements-for-weights", llvm::cl::init(0),
      llvm::cl::desc("The minimum required number of elements in a weight "
                     "array to apply quantization.")};

  Option<QuantMethod> quantization_method_{
      *this, "quantization-method",
      llvm::cl::init(tensorflow::quantization::QuantizationMethod::
                         METHOD_DYNAMIC_RANGE_INT8),
      llvm::cl::desc("Choose quantization method."),
      llvm::cl::values(
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_DYNAMIC_RANGE_INT8,
                     "drq", "Post-training dynamic-range quantizaiton"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8,
                     "weight_only", "Post-training weight_only quantizaiton"))};
};

class CheckQuantizableOps
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit CheckQuantizableOps(MLIRContext* context,
                               const QuantMethod quantization_method,
                               const OpSet target_opset,
                               const int min_num_elements_for_weights)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        quantization_method_(quantization_method),
        target_opset_(target_opset),
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

    StringRef function_name =
        call_op.getFAttr().cast<FlatSymbolRefAttr>().getValue();
    if ((quantization_method_ == tensorflow::quantization::QuantizationMethod::
                                     METHOD_DYNAMIC_RANGE_INT8) &&
        (function_name.contains("batch_matmul") ||
         function_name.contains("conv3d"))) {
      call_op->removeAttr(kQuantTraitAttrName);
    }

    // TODO(b/270906404): Support weight-only gather for uniform quantized opset
    // in PTQ mode
    if (target_opset_ == OpSet::UNIFORM_QUANTIZED &&
        function_name.contains("gather")) {
      call_op->removeAttr(kQuantTraitAttrName);
    }

    return failure();
  }
  QuantMethod quantization_method_;
  OpSet target_opset_;
  int min_num_elements_for_weights_;
};

static PassRegistration<LiftQuantizableSpotsAsFunctionsDRQPass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/lift_quantizable_spots_as_functions_drq.inc"

void LiftQuantizableSpotsAsFunctionsDRQPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  populateWithGenerated(patterns);
  patterns.add<CheckQuantizableOps>(ctx, quantization_method_, target_opset_,
                                    min_num_elements_for_weights_);
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
CreateLiftQuantizableSpotsAsFunctionsDRQPass(
    const QuantMethod quantization_method, const OpSet target_opset,
    const int min_num_elements_for_weights) {
  return std::make_unique<LiftQuantizableSpotsAsFunctionsDRQPass>(
      quantization_method, target_opset, min_num_elements_for_weights);
}

}  // namespace quant
}  // namespace mlir

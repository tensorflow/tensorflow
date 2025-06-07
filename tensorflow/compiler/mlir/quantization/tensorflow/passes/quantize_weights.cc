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
#include <optional>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/temp_tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_quantize_op.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_quant::GetWeightComponentSpec;
using ::mlir::tf_quant::IsOpWithDataMovementTrait;
using ::mlir::tf_quant::IsOpWithQuantizableTrait;
using ::mlir::tf_quant::IsValueWithQuantizablePrecision;

class QuantizeWeightsPass
    : public mlir::PassWrapper<QuantizeWeightsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeWeightsPass)

  explicit QuantizeWeightsPass() : test_mode_(true) { initializeForTest(); }

  explicit QuantizeWeightsPass(
      const tensorflow::quantization::QuantizationOptions& quant_options)
      : test_mode_(false), quant_options_(quant_options) {}

  QuantizeWeightsPass(const QuantizeWeightsPass& other) {
    test_mode_ = other.test_mode_;
    quant_options_ = other.quant_options_;
    initializeForTest();
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-quantize-weights";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Quantize weights used by quantizable ops.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, quant::QuantDialect>();
  }

 private:
  void runOnOperation() override;

  bool test_mode_;
  tensorflow::quantization::QuantizationOptions quant_options_;

  // Initialize for tests.
  void initializeForTest() {
    if (!test_mode_) return;

    tensorflow::quantization::QuantizationComponentSpec quant_spec;
    quant_spec.set_quantization_component(
        tensorflow::quantization::QuantizationComponentSpec::COMPONENT_WEIGHT);
    quant_spec.set_tensor_type(
        tensorflow::quantization::QuantizationComponentSpec::TENSORTYPE_INT_8);
    auto mutable_quant_method = quant_options_.mutable_quantization_method();
    *mutable_quant_method->add_quantization_component_specs() = quant_spec;
  }
};

// If a constant is connected to a quantizable op, quantize the constant to have
// the provided data type.
class QuantizeConstWeights : public OpRewritePattern<TF::ConstOp> {
 public:
  explicit QuantizeConstWeights(
      MLIRContext* context,
      const tensorflow::quantization::QuantizationOptions& quantization_options)
      : OpRewritePattern<TF::ConstOp>(context),
        quant_options_(quantization_options) {}

  LogicalResult matchAndRewrite(TF::ConstOp op,
                                PatternRewriter& rewriter) const override {
    auto weight_component_spec = GetWeightComponentSpec(quant_options_);
    if (!weight_component_spec) return failure();

    // 1. Check if the constant is quantizable.
    if (failed((isQuantizableWeight(op)))) {
      return failure();
    }

    // 2. Quantize the constant to the provided data type.
    // After quantization, the graph will be transformed
    // from:
    // const -> some op -> quantizable_op
    // to:
    // q_const -> dequant_op -> some op -> quantizable_op
    //
    // A dequant_op will propagate to further quantize the next ops in another
    // pass.
    //
    // Note that a constant can be used by multiple ops. For example, if a graph
    // looks like below:
    // const -> while -> quant_op
    //       -> not_quant_op
    //
    // the transformation will be:
    // q_const -> dequant_op -> while -> quant_op
    //                       -> not_quant_op
    // And the dequant_op op will propagate towards quant_op only.
    if (failed(quantizeOps(rewriter, op, weight_component_spec.value()))) {
      return failure();
    }
    return success();
  }

 private:
  // Check if op's user or op's user after an identity op is connected to a
  // terminator.
  bool checkIfAnyUserIsConnectedToTermiantor(BlockArgument op) const {
    for (const auto& user : op.getUsers()) {
      if (user->template hasTrait<OpTrait::IsTerminator>()) return true;
      if (auto next_user = dyn_cast_or_null<TF::IdentityOp>(user)) {
        return (*(next_user->getResult(0).getUsers().begin()))
            ->template hasTrait<OpTrait::IsTerminator>();
      }
    }
    return false;
  }

  // Check if the constant op is connected to a quantizable op at some point.
  bool hasUsageFromQuantizableOp(TF::ConstOp op) const {
    llvm::SmallVector<mlir::Value> uses_at_current_level{op};
    while (!uses_at_current_level.empty()) {
      llvm::SmallVector<mlir::Value> next_values_to_visit;
      for (auto cur_op : uses_at_current_level) {
        for (auto& cur_op_use : cur_op.getUses()) {
          Operation* next_op = cur_op_use.getOwner();
          int next_op_operand_num = cur_op_use.getOperandNumber();
          if (auto call_op = llvm::dyn_cast<mlir::CallOpInterface>(next_op)) {
            mlir::func::FuncOp func =
                llvm::dyn_cast<mlir::func::FuncOp>(call_op.resolveCallable());
            if (!func) continue;
            next_values_to_visit.push_back(
                func.getArgument(next_op_operand_num));
          } else if (auto while_op =
                         llvm::dyn_cast_or_null<TF::WhileOp>(next_op)) {
            func::FuncOp func = while_op.body_function();
            auto func_argument = func.getArgument(next_op_operand_num);
            // Check if the op is returned without mutation. Returning values
            // from a while op follow return or identity -> return pattern.
            if (checkIfAnyUserIsConnectedToTermiantor(func_argument))
              next_values_to_visit.push_back(
                  func.getArgument(next_op_operand_num));
          } else if (IsOpWithQuantizableTrait(next_op)) {
            // Check this before IsOpWithDataMovementTrait since some data
            // movement ops are also quantizable ops.
            return true;
          } else if (IsOpWithDataMovementTrait(next_op)) {
            next_values_to_visit.insert(next_values_to_visit.end(),
                                        next_op->getResults().begin(),
                                        next_op->getResults().end());
          }
        }
      }
      uses_at_current_level.swap(next_values_to_visit);
    }
    return false;
  }

  // List of conditions to check if a const op is quantizable.
  LogicalResult isQuantizableWeight(TF::ConstOp op) const {
    // Non-float tensors do not need quantization.
    if (!IsValueWithQuantizablePrecision(op)) return failure();
    // Check if quantizable ops are connected. Do this before num_elements check
    // to avoid checking unnecessary constants which causes unintended remarks.
    // This check also prevents quantizing unintended consts like scale.
    if (!hasUsageFromQuantizableOp(op)) return failure();

    // Check if the weight size is big enough.
    int num_elements_threshold = quant_options_.min_num_elements_for_weights();
    int num_elements = cast<ShapedType>(op.getType()).getNumElements();
    if (num_elements < num_elements_threshold) {
      op->emitRemark("Quantization is skipped because the op has ")
          << num_elements << " elements which is fewer than the threshold("
          << num_elements_threshold << " elements).";
      return failure();
    }

    return success();
  }

  // Apply quantization with the provided spec.
  LogicalResult quantizeOps(PatternRewriter& rewriter, TF::ConstOp op,
                            tensorflow::quantization::QuantizationComponentSpec&
                                weight_component_spec) const {
    if (weight_component_spec.tensor_type() ==
        tensorflow::quantization::QuantizationComponentSpec::TENSORTYPE_INT_8) {
      // TODO - b/296535985: [Converter Component][TF-Quantizer] Factor out
      // quant/dequant in QuantizeWeightsPass
      auto dequantized_val =
          ApplyUniformQuantization(rewriter, op, weight_component_spec);
      if (!dequantized_val.has_value()) return failure();
      op.getOutput().replaceAllUsesWith(dequantized_val.value().getResult(0));
      return success();
    }

    op->emitRemark("Not supported quantization data type.");
    return failure();
  }

 protected:
  tensorflow::quantization::QuantizationOptions quant_options_;
};

static PassRegistration<QuantizeWeightsPass> pass;

void QuantizeWeightsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  auto module_op = getOperation();
  RewritePatternSet patterns(ctx);

  patterns.add<QuantizeConstWeights>(ctx, quant_options_);

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  // Apply transformation on each function. For recursive call case, another
  // function can be modified at the same time so avoid running functions in
  // parallel.
  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-quantize-weights failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeWeightsPass(
    const tensorflow::quantization::QuantizationOptions& quant_options) {
  return std::make_unique<QuantizeWeightsPass>(quant_options);
}

}  // namespace quant
}  // namespace mlir

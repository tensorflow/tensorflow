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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_QUANTIZATION_PATTERNS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_QUANTIZATION_PATTERNS_H_

#include <type_traits>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/ops/stablehlo_op_quant_spec.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir::quant::stablehlo {

// Checks whether an op is connected with a quantized composite function. If
// not, the same-scale op will not be quantized. This decision is based on the
// current assumption that the performance gain of the same-scale op itself
// could not beat the overhead of the quantize and dequantize routines need to
// be added around that op. When the assumption changes, this policy might
// change as well.
bool IsConnectedWithQuantizedCompsiteFunction(Operation* same_scale_op);

// A base rewrite pattern which matches any N-in-M-out operations with
// quantization parameters propagated to at least one of its operands. The
// quantization parameters are annotated by the QuantizeOp/DequantizeOp pairs.
// Each matched pattern are rewritten by its quantized alternatives.
//
// Quantization method is determined by the `_quantization_method` attributes
// attached to each quantizable units.
//
// Template constraints are imposed as follows:
//
// * `QuantizeOpT` should have only one operand.
// * `DequantizeOpT` should have only one result.
template <typename ConcreteT, typename QuantizeOpT, typename DequantizeOpT,
          typename VerifierT, typename RootOpT = DequantizeOpT,
          typename = std::enable_if_t<
              QuantizeOpT::template hasTrait<OpTrait::OneOperand>() &&
              DequantizeOpT::template hasTrait<OpTrait::OneResult>()>>
class StableHloQuantizationPattern : public OpRewritePattern<RootOpT> {
 public:
  explicit StableHloQuantizationPattern(MLIRContext* context)
      // Set the benefit to a large number so that it is always preferred.
      : OpRewritePattern<RootOpT>(context, /*benefit=*/300) {}

 private:
  // Collects all candidate ops for quantization, which are the
  // `dequantize_op`'s users.
  FailureOr<SmallVector<Operation*>> CollectCandidateOps(
      DequantizeOpT dequantize_op) const {
    auto users = dequantize_op->getResult(0).getUsers();
    return SmallVector<Operation*>(users.begin(), users.end());
  }

  // Collects all candidate ops for quantization, which is the operand of
  // `quantize_op`. If successful, this always returns one element which is the
  // operand of `quantize_op`.
  FailureOr<SmallVector<Operation*>> CollectCandidateOps(
      QuantizeOpT quantize_op) const {
    Value operand = quantize_op->getOperand(0);
    if (QuantizedType::getQuantizedElementType(operand.getType())) {
      // The input of the quantize op has already been quantized, i.e.
      // rescale.
      return failure();
    }

    Operation* operand_op = operand.getDefiningOp();
    if (operand_op == nullptr) {
      // When `QuantizeOpT`'s operand does not have a defining op, it means it
      // is a `BlockArgument`. The pattern does not match if there is no op to
      // quantize.
      return failure();
    }

    if (operand_op->hasTrait<OpTrait::ConstantLike>()) {
      // Const-> QuantizeOp pattern will be handled separately.
      return failure();
    }

    return SmallVector<Operation*>{operand_op};
  }

  LogicalResult matchAndRewrite(RootOpT op,
                                PatternRewriter& rewriter) const override {
    // Collect all the candidate ops for quantization.
    FailureOr<SmallVector<Operation*>> candidate_ops = CollectCandidateOps(op);
    // Safeguard check to ensure that there is at least one quantizable op.
    if (failed(candidate_ops) || candidate_ops->empty()) return failure();

    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (Operation* candidate_op : *candidate_ops) {
      // If it is requantize op, we shouldn't rewrite this op.
      if (isa<QuantizeOpT, DequantizeOpT>(candidate_op)) {
        return failure();
      }

      // If the op is terminator, we shouldn't rewrite.
      if (candidate_op->hasTrait<OpTrait::IsTerminator>()) {
        return failure();
      }

      if (!IsOpQuantizableStableHlo(candidate_op)) {
        return failure();
      }

      if (GetStableHloQuantConstraints(candidate_op)
              ->has_same_scale_requirement &&
          !IsConnectedWithQuantizedCompsiteFunction(candidate_op)) {
        return failure();
      }

      // Ops with regions will be quantized in a separate pattern.
      if (isa<mlir::stablehlo::ReduceWindowOp>(candidate_op)) {
        return failure();
      }

      const bool weight_only_quantizable =
          IsWeightOnlyQuantizableOp(*candidate_op);

      // Collect all the quantized inputs and "clone" the matched op by these
      // inputs.
      SmallVector<Value, 4> inputs;
      inputs.reserve(candidate_op->getNumOperands());
      for (auto operand : candidate_op->getOperands()) {
        Type operand_type = operand.getType();
        if (mlir::isa<NoneType>(operand_type)) {
          inputs.push_back(operand);
          continue;
        }

        auto ele_type =
            mlir::cast<TensorType>(operand.getType()).getElementType();
        if (auto dq_op =
                dyn_cast_or_null<DequantizeOpT>(operand.getDefiningOp())) {
          inputs.push_back(dq_op.getOperand());
        } else if (!ele_type.isF32()) {
          // If the operand is an integer tensor, then it doesn't require the
          // DequantizeOp in the pattern.
          inputs.push_back(operand);
        } else if (weight_only_quantizable) {
          inputs.push_back(operand);
        } else {
          return failure();
        }
      }

      // Collect all the quantized outputs and replace them by the results of
      // the new quantized op.
      llvm::SmallDenseMap<Value, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(candidate_op->getNumResults());
      for (const auto& enumerated_result :
           llvm::enumerate(candidate_op->getResults())) {
        Value result = enumerated_result.value();
        Type result_type = result.getType();
        // Add this to the test coverage once we create test ops with none type
        // results.
        if (mlir::isa<NoneType>(result_type)) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result_type);
          continue;
        }
        Type result_ele_type =
            mlir::cast<TensorType>(result.getType()).getElementType();
        // If the user is the QuantizeOp, it must be the only user.
        if (result.hasOneUse() && isa<QuantizeOpT>(*result.user_begin())) {
          auto user = cast<QuantizeOpT>(*result.user_begin());
          outputs_replaced.insert(
              {user.getResult(), enumerated_result.index()});
          output_types.push_back(user.getType());
        } else if (!result_ele_type.isF32()) {
          // If the result is an integer tensor, then it doesn't require the
          // D op in the pattern.
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else if (weight_only_quantizable) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else {
          return failure();
        }
      }

      rewriter.setInsertionPointAfter(candidate_op);
      OperationState new_state(candidate_op->getLoc(),
                               candidate_op->getName().getStringRef(), inputs,
                               output_types, candidate_op->getAttrs());
      for (int i = 0; i < candidate_op->getNumRegions(); ++i) {
        new_state.addRegion();
      }
      Operation* quantized_op = rewriter.create(new_state);
      if (candidate_op->getNumRegions() != 0) {
        for (const auto& indexed_regions :
             llvm::enumerate(candidate_op->getRegions())) {
          Region& target_region =
              quantized_op->getRegion(indexed_regions.index());
          IRMapping mapping;
          indexed_regions.value().cloneInto(&target_region, mapping);
        }
      }
      for (auto output : outputs_replaced) {
        output.getFirst().replaceAllUsesWith(
            quantized_op->getResult(output.getSecond()));
      }
    }
    return success();
  }
};

// Populates common patterns that are usually compute heavy or memory bound.
void PopulateCommonQuantizationPatterns(
    MLIRContext& ctx, RewritePatternSet& patterns,
    bool enable_per_channel_quantized_weight);

// Populates conversion patterns for all quantizable ops, including
// ops that are not compute-heavy and data movement ops.
void PopulateAllQuantizablePatterns(MLIRContext& ctx,
                                    RewritePatternSet& patterns);

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_QUANTIZATION_PATTERNS_H_

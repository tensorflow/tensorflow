/* Copyright 2023 The StableHLO Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/fill_quantization_options.h"

// NOLINTNEXTLINE
//===----------------------------------------------------------------------===//
// The Quantization Pass for Weight.
//===----------------------------------------------------------------------===//

namespace mlir::quant::stablehlo {

// Put the definitions inside the ::mlir::quant::stablehlo namespace, to match
// the declarations in passes.h.
#define GEN_PASS_DEF_QUANTIZEWEIGHTPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using QuantizationUnits = llvm::SetVector<std::pair<Operation*, int>>;
using mlir::stablehlo::ConstantOp;
using mlir::stablehlo::ConvertOp;
using ::stablehlo::quantization::QuantizationComponentSpec;

// Min/Max values used for creating ConstantOp.
constexpr float kMaxFloat16Value = 65504.f;
constexpr float kMinFloat16Value = -65504.f;

class QuantizeWeightPass
    : public impl::QuantizeWeightPassBase<QuantizeWeightPass> {
 public:
  explicit QuantizeWeightPass(
      QuantizationComponentSpec quantization_component_spec)
      : quantization_component_spec_(quantization_component_spec) {}

 private:
  void runOnOperation() override;
  QuantizationComponentSpec quantization_component_spec_;
};

// Collects quantizable target ops, then insert Q-DQ quantization patterns.
class QuantizeWeight : public OpRewritePattern<ConstantOp> {
 public:
  explicit QuantizeWeight(
      MLIRContext* context,
      const QuantizationComponentSpec& quantization_component_spec)
      : OpRewritePattern<ConstantOp>(context),
        quantization_component_spec_(quantization_component_spec) {}

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter& rewriter) const override {
    // 1. Collect quantizable ops.
    QuantizationUnits quantizable_ops = GetQuantizableOps(op);
    if (quantizable_ops.empty()) {
      return failure();
    }

    // 2. Quantize collected ops.
    if (!QuantizeOps(rewriter, op, quantizable_ops)) {
      return failure();
    }

    // 3. Complete the Q-DQ pair for each inference type.
    if (!ConvertToFloat16Constant(rewriter, op)) {
      return failure();
    }
    return success();
  }

 private:
  const QuantizationComponentSpec quantization_component_spec_;
  // Marks users that are applicable for quantization where the criteria for
  // determining quantizable ops differs by the inference type.
  QuantizationUnits GetQuantizableOps(ConstantOp op) const {
    // Non-float tensors do not need quantization.
    QuantizationUnits quantizable_ops;
    const ShapedType type = mlir::dyn_cast<ShapedType>(op.getType());
    if (!type || !type.getElementType().isF32()) return quantizable_ops;

    const Value value = op.getResult();

    for (OpOperand& use : value.getUses()) {
      Operation* user = use.getOwner();
      const int operand_num = use.getOperandNumber();
      quantizable_ops.insert({user, operand_num});
    }
    return quantizable_ops;
  }

  // Returns whether quantization is applied to filtered users.
  bool QuantizeOps(PatternRewriter& rewriter, ConstantOp op,
                   const QuantizationUnits& quantizable_ops) const {
    for (const std::pair<Operation*, int>& quant_op : quantizable_ops) {
      // For f16 quantization, quantize all constant ops as float16.
      QuantizeOpAsFloat16(rewriter, op, quant_op);
    }
    // TODO: b/264218457 - Return a value that accurately captures result
    // status.
    return true;
  }

  // Inserts ConvertOp which is used for converting float32 ConstantOp into
  // float16 quantization. If there is an existing ConvertOp connected to the
  // ConstantOp, the quantizable_op will be rewired to the existing ConvertOp.
  // This guarantees at most one ConvertOp is created for float32 to float16
  // conversion.
  void QuantizeOpAsFloat16(PatternRewriter& rewriter, ConstantOp op,
                           const std::pair<Operation*, int> quant_op) const {
    const auto [quantizable_op, quantize_operand_num] = quant_op;
    // If the constant is an output tensor, do nothing.
    if (isa<func::ReturnOp>(quantizable_op)) {
      return;
    }

    TensorType old_result_type =
        mlir::dyn_cast<TensorType>(op.getResult().getType());
    const FloatType quantized_type = Float16Type::get(op.getContext());
    const ShapedType new_result_type = old_result_type.clone(quantized_type);

    // Insert ConvertOp if it does not exist yet. Otherwise, just rewire without
    // creating a ConvertOp.
    for (const OpOperand& connected_op : op.getResult().getUses()) {
      ConvertOp convert_op =
          dyn_cast_or_null<ConvertOp>(connected_op.getOwner());
      // ConvertOp already exists. Rewire the existing convert op into f16.
      if (convert_op && convert_op.getType() == new_result_type) {
        quantizable_op->setOperand(quantize_operand_num, convert_op);
        return;
      }
    }
    rewriter.setInsertionPointAfter(op);
    ConvertOp new_convert_op = rewriter.create<ConvertOp>(
        op->getLoc(), new_result_type, op.getResult());
    quantizable_op->setOperand(quantize_operand_num,
                               new_convert_op.getResult());
  }

  // Returns whether a ConvertOp-Operation sequence can be converted into new
  // ConstantOp-Convert-Operation. The new ConstantOp has float16 data type.
  bool ConvertToFloat16Constant(PatternRewriter& rewriter,
                                ConstantOp op) const {
    for (Operation* connected_op : op.getResult().getUsers()) {
      ConvertOp convert_op = dyn_cast_or_null<ConvertOp>(connected_op);
      // Skip if no convert op exists.
      if (!convert_op || convert_op.getResult().use_empty()) continue;

      // Get types.
      const Type old_result_type = op.getResult().getType();
      const ShapedType new_result_type =
          mlir::dyn_cast<ShapedType>(convert_op.getType());

      // Proceeds only if the converting is to float16.
      if (!new_result_type.getElementType().isF16()) continue;

      // Convert values.
      std::vector<Eigen::half> new_values;
      const DenseFPElementsAttr value_attr =
          mlir::cast<DenseFPElementsAttr>(op.getValue());
      new_values.reserve(value_attr.getNumElements());

      for (const float value : value_attr.getValues<float>()) {
        new_values.push_back(Eigen::half(
            std::min(std::max(value, kMinFloat16Value), kMaxFloat16Value)));
      }
      const DenseElementsAttr new_value_attr = DenseFPElementsAttr::get(
          new_result_type, ArrayRef<Eigen::half>(new_values));
      // Create new ConstantOp-ConvertOp-Operation sequences. At this moment,
      // old ConstantOp is guaranteed to have one F32->F16 convert op regardless
      // of its number of users.
      rewriter.setInsertionPointAfter(op);
      // create new F16 constant op in that location
      ConstantOp new_const = rewriter.create<ConstantOp>(
          op->getLoc(), new_result_type, new_value_attr);
      ConvertOp dcast =
          rewriter.create<ConvertOp>(op->getLoc(), old_result_type, new_const);
      // replace all convert ops with dq op.
      convert_op->replaceAllUsesWith(dcast);
      // Return without scanning for the next ConvertOp as only one ConvertOp is
      // connected to all quantizable ops.
      return true;
    }
    return false;
  }
};

// TODO: b/264218457 - Refactors the current file to parse preset quantization
// options and allow modular control of quantization specs.
void QuantizeWeightPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<QuantizeWeight>(ctx, quantization_component_spec_);

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  if (failed(applyPatternsGreedily(func, frozen_patterns))) {
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the StableHLO dialect Quantize Weight pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizeWeightPass(
    const QuantizationComponentSpec& quantization_component_spec) {
  return std::make_unique<QuantizeWeightPass>(quantization_component_spec);
}

}  // namespace mlir::quant::stablehlo

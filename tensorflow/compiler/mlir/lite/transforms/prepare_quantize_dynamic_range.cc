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

#include <algorithm>
#include <cstdint>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

// NOLINTNEXTLINE
//===----------------------------------------------------------------------===//
// The prepare-dynamic-range-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// A boolean attribute used to describe whether input activations need to be
// asymmetrically quantized.
constexpr char kAsymmetricQuantizeInputsAttr[] = "asymmetric_quantize_inputs";

using QuantizationUnits = llvm::SetVector<std::pair<Operation*, int>>;

// Applies prepare dynamic range quantization on the model in TFL dialect.
// This pass runs before the quantization pass and apply preprocess if
// applicable.
class PrepareDynamicRangeQuantizePass
    : public PrepareDynamicRangeQuantizePassBase<
          PrepareDynamicRangeQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareDynamicRangeQuantizePass)

  // Constructor used by the PassRegistration. This is only used by test.
  explicit PrepareDynamicRangeQuantizePass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
    quant_specs_.weight_quantization = true;
    quant_specs_.enable_mlir_dynamic_range_quantizer = true;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareDynamicRangeQuantizePass(
      const quant::QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {
    enable_dynamic_range_per_channel_quantization_ =
        !quant_specs_.disable_per_channel;
    min_elements_for_weights_ = quant_specs_.minimum_elements_for_weights;
  }

  // The function might contain stats ops which are redundant for processing
  // dynamic range quantization. And stats ops may cause conflict while
  // processing the function for dynamic range quantization. Therefore, this
  // method preprocess the function to remove all stats ops.
  void removeAllStatsOp(func::FuncOp func);

  void runOnOperation() override;

 private:
  quant::QuantizationSpecs quant_specs_;
};

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

// If the weight is applicable to dynamic range quantization, insert Quantize
// and Dequantize ops with either per-axis or per-tensor scale.
class PrepareDynamicRangeQuantizableOp
    : public OpRewritePattern<arith::ConstantOp> {
 public:
  explicit PrepareDynamicRangeQuantizableOp(
      MLIRContext* context, const quant::QuantizationSpecs& quant_specs)
      : OpRewritePattern<arith::ConstantOp>(context),
        quant_specs_(quant_specs) {}

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    QuantizationUnits quantizable_ops;

    // 1. Collect quantizable ops.
    if (!(getQuantizableOps(op, quantizable_ops))) {
      return failure();
    }

    // 2. Quantize collected ops. It is immediatly quantized by inserting Q-DQ
    // pair for int8 while it is lazily applied for float16 by inserting CastOp.
    if (!(quantizeOps(rewriter, op, quantizable_ops))) {
      return failure();
    }

    // 3. Apply post-processing required for each inference type.
    // TODO(b/212514817): refactor mode checking to improve code quality
    if (quant_specs_.inference_type == tensorflow::DT_QINT8 &&
        (setAsymmetricQuantizeInputAttr(rewriter, quantizable_ops))) {
      return failure();
    }
    if (quant_specs_.inference_type == tensorflow::DT_HALF &&
        (convertToFloat16Constant(rewriter, op))) {
      return failure();
    }

    return success();
  }

 private:
  // Check if the operand_index is included in the quantizable_indices.
  bool isQuantizableIndex(const int operand_index,
                          const std::vector<int>& quantizable_indices) const {
    return std::find(std::begin(quantizable_indices),
                     std::end(quantizable_indices),
                     operand_index) != std::end(quantizable_indices);
  }

  // Check if any specific operand and its index pair is supported for int8
  // quantization. For dynamic range quantizable ops, it refers to the op
  // specification for checking the support. For custom ops, it checks the
  // provided map.
  bool hasInt8QuantizableOperandAt(Operation* op, int operand_index) const {
    if (auto custom_op = llvm::dyn_cast_or_null<CustomOp>(op)) {
      std::string op_name = custom_op.custom_code().str();
      auto custom_map_iter = quant_specs_.custom_map.find(op_name);
      if (custom_map_iter != quant_specs_.custom_map.end())
        return isQuantizableIndex(
            operand_index, custom_map_iter->second.quantizable_input_indices);
    } else if (auto quantizable_op =
                   llvm::dyn_cast<DynamicRangeQuantizedOpInterface>(op)) {
      const auto& quantizable_indices =
          quantizable_op.GetQuantizableOperandIndices();
      return isQuantizableIndex(operand_index, quantizable_indices);
    }
    return false;
  }

  // Insert CastOp which is used to for converting float32 ConstantOp into
  // float16 quantization. If there is an existing CastOp connected to the
  // ConstantOp, the quantize_op will be rewired to the existing CastOp. This
  // guarentees at most one CastOp is created for float32 to float16 conversion.
  void quantizeOpAsFloat16(PatternRewriter& rewriter, arith::ConstantOp op,
                           std::pair<Operation*, int> quant_op) const {
    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    // If the constant is an output tensor, do nothing.
    if (llvm::dyn_cast_or_null<func::ReturnOp>(quantize_op)) {
      return;
    }

    // Get types
    TensorType old_result_type =
        op.getResult().getType().template dyn_cast<TensorType>();
    FloatType quantized_type = FloatType::getF16(op.getContext());
    ShapedType new_result_type = old_result_type.clone(quantized_type);

    // Insert CastOp if it does not exist yet. Otherwise, just rewire without
    // creating a CastOp.
    for (auto& connected_op : op.getResult().getUses()) {
      auto cast_op = llvm::dyn_cast_or_null<CastOp>(connected_op.getOwner());
      if (cast_op && cast_op.getType() == new_result_type) {
        quantize_op->setOperand(quantize_operand_num, cast_op);
        return;
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto new_cast_op =
        rewriter.create<CastOp>(op->getLoc(), new_result_type, op.getResult());
    quantize_op->setOperand(quantize_operand_num, new_cast_op.getResult());
  }

  // Apply per-axis quantization if applicable. Otherwise, apply per-tensor
  // quantization for int8 dynamic range quantization.
  bool quantizeOpAsInt8(PatternRewriter& rewriter, arith::ConstantOp op,
                        std::pair<Operation*, int> quant_op) const {
    bool is_narrow_range = true;
    bool is_legacy_float = quant_specs_.legacy_float_scale;
    bool is_signed = quant_specs_.IsSignedInferenceType();
    int bit_width = quant_specs_.GetQuantizationTypeWidth();

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    auto affine_user = dyn_cast<AffineQuantizedOpInterface>(quantize_op);

    bool op_with_per_axis_support = false;

    if (!llvm::dyn_cast_or_null<CustomOp>(quantize_op)) {
      bool op_with_narrow_range =
          affine_user &&
          affine_user.GetAffineOperandIndex() == quantize_operand_num &&
          affine_user.RequiredNarrowRangeAffineOperand();

      op_with_per_axis_support = op_with_narrow_range &&
                                 affine_user.GetQuantizationDimIndex() != -1 &&
                                 !quant_specs_.disable_per_channel;
    }

    QuantizedType quant_type = nullptr;
    DenseFPElementsAttr attr;
    if (!matchPattern(op->getResult(0), m_Constant(&attr))) return false;

    if (attr.dyn_cast<DenseFPElementsAttr>().size() <
        quant_specs_.minimum_elements_for_weights) {
      op->emitRemark("Quantization is skipped for ")
          << quantize_op->getName().getStringRef().str() << " because it has "
          << attr.dyn_cast<DenseFPElementsAttr>().size()
          << " elements which is fewer than the threshold("
          << quant_specs_.minimum_elements_for_weights << " elements).";
      return false;
    }

    if (op_with_per_axis_support) {
      quant_type = quant::GetUniformQuantizedPerAxisTypeForWeight(
                       attr, affine_user.GetQuantizationDimIndex(),
                       /*symmetric=*/true, bit_width, is_signed,
                       is_narrow_range, is_legacy_float)
                       .template dyn_cast<quant::QuantizedType>();
    } else {
      quant_type = quant::GetUniformQuantizedTypeForWeight(
                       attr, is_narrow_range && is_signed, bit_width, is_signed,
                       is_narrow_range, is_legacy_float)
                       .template dyn_cast<quant::QuantizedType>();
    }
    return insertQDQ(rewriter, op, quant_type, quant_op);
  }

  // Insert Quantize and Dequantize ops.
  bool insertQDQ(PatternRewriter& rewriter, arith::ConstantOp op,
                 QuantizedType quant_type,
                 std::pair<Operation*, int> quant_op) const {
    if (!quant_type) return false;

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    Type expressed_type = op.getResult().getType();
    Type cast_type = quant_type.castFromExpressedType(expressed_type);

    // Insert DQ-op if it does not exist yet. Otherwise, just rewire without
    // creating a new DQ-op.
    for (auto connected_op : op->getUsers()) {
      auto q_op = llvm::dyn_cast_or_null<Q>(connected_op);
      if (q_op && q_op.getType() == cast_type) {
        auto dq_op = llvm::cast<DQ>(q_op.getResult().use_begin()->getOwner());
        quantize_op->setOperand(quantize_operand_num, dq_op);
        return false;
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<Q>(op->getLoc(), cast_type, op.getResult());
    auto dq = rewriter.create<DQ>(op->getLoc(), expressed_type, q);
    quantize_op->setOperand(quantize_operand_num, dq.getResult());
    return true;
  }

  // Mark users that are applicable for dynamic range quantization where the
  // criteria for determining quantizable ops differs by the inferentce type.
  bool getQuantizableOps(arith::ConstantOp op,
                         QuantizationUnits& quantizable_ops) const {
    // Non-float tensors do not need quantization.
    auto type = op.getType().dyn_cast<ShapedType>();
    if (!type || !type.getElementType().isF32()) return false;

    Value value = op.getResult();

    // Check whether dynamic range quantization can be applied.
    for (auto& use : value.getUses()) {
      Operation* user = use.getOwner();
      int operand_num = use.getOperandNumber();

      // TODO(b/212514817): refactor mode checking to improve code quality
      if (quant_specs_.inference_type == tensorflow::DT_QINT8 &&
          hasInt8QuantizableOperandAt(user, operand_num)) {
        quantizable_ops.insert({user, operand_num});
      } else if (quant_specs_.inference_type == tensorflow::DT_HALF) {
        quantizable_ops.insert({user, operand_num});
      }
    }
    return !quantizable_ops.empty();
  }

  // For each filtered user, apply quantization.
  bool quantizeOps(PatternRewriter& rewriter, arith::ConstantOp op,
                   QuantizationUnits& quantizable_ops) const {
    bool quantized = false;

    // TODO(b/212514817): refactor mode checking to improve code quality
    for (auto& quant_op : quantizable_ops) {
      if (quant_specs_.inference_type == tensorflow::DT_QINT8) {
        quantized |= quantizeOpAsInt8(rewriter, op, quant_op);
      } else if (quant_specs_.inference_type == tensorflow::DT_HALF) {
        quantizeOpAsFloat16(rewriter, op, quant_op);
        quantized = true;
      }
    }
    return quantized;
  }

  // Add asymmetric input quantization attribute. MLIR dynamic quantization
  // supports only the case that the value of the attribute equals to true. For
  // details, see tensorflow/compiler/mlir/lite/quantization/quantization.td
  bool setAsymmetricQuantizeInputAttr(
      PatternRewriter& rewriter, QuantizationUnits& quantizable_ops) const {
    bool changed = false;
    for (auto& quant_op : quantizable_ops) {
      auto dynamic_range_quantized_user =
          dyn_cast<DynamicRangeQuantizedOpInterface>(quant_op.first);
      if (dynamic_range_quantized_user &&
          dynamic_range_quantized_user.RequireAsymmetricQuantizeInputsAttr()) {
        // At runtime, this flag will be used in the kernels to decide whether
        // input activations need to be asymmetrically quantized. Refer to the
        // implementation for fully-connected as an example in
        // tensorflow/lite/kernels/fully_connected.cc. The kernels will handle
        // the asymmetric_quantize_inputs attribute in the builtin option.
        dynamic_range_quantized_user->setAttr(
            kAsymmetricQuantizeInputsAttr,
            BoolAttr::get(rewriter.getContext(), true));
        changed = true;
      }
    }
    return changed;
  }

  // Convert ConstantOp-CastOp-Operation sequence into new ConstantOp
  // -Dequantize-Operation where the new ConstantOp has float16 data type.
  bool convertToFloat16Constant(PatternRewriter& rewriter,
                                arith::ConstantOp op) const {
    for (auto connected_op : op.getResult().getUsers()) {
      auto cast_op = dyn_cast_or_null<CastOp>(connected_op);
      if (!cast_op || cast_op.getResult().use_empty()) continue;

      // Get types
      Type old_result_type = op.getResult().getType();
      ShapedType new_result_type =
          cast_op.getType().template dyn_cast<ShapedType>();

      // Proceeds only if the casting is to float16
      if (!new_result_type.getElementType().isF16()) continue;

      // Cast values
      std::vector<Eigen::half> new_values;
      DenseFPElementsAttr value_attr =
          op.getValue().cast<DenseFPElementsAttr>();
      new_values.reserve(value_attr.getNumElements());

      constexpr float kMaxFloat16Value = 65504.f;
      constexpr float kMinFloat16Value = -65504.f;

      for (auto value : value_attr.template getValues<float>()) {
        new_values.push_back(Eigen::half(
            std::min(std::max(value, kMinFloat16Value), kMaxFloat16Value)));
      }
      DenseElementsAttr new_value_attr = DenseFPElementsAttr::get(
          new_result_type, ArrayRef<Eigen::half>(new_values));

      // Create new ConstantOp-Dequantize-Operation sequences. At this moment,
      // old ConstantOp is guaranteed to have one F32->F16 cast regardless of
      // its number of users.
      rewriter.setInsertionPointAfter(op);
      auto new_const = rewriter.create<arith::ConstantOp>(
          op->getLoc(), new_result_type, new_value_attr);
      auto dq = rewriter.create<DQ>(op->getLoc(), old_result_type, new_const);
      cast_op->replaceAllUsesWith(dq);

      // Return without scanning for the next CastOp as only one CastOp is
      // connected to all quantizable ops.
      return true;
    }
    return false;
  }

 protected:
  quant::QuantizationSpecs quant_specs_;
};

// Remove all the stats ops which are redundant for dynamic range quantizaiton.
void PrepareDynamicRangeQuantizePass::removeAllStatsOp(func::FuncOp func) {
  func.walk([&](quant::StatisticsOp stats_op) {
    stats_op.replaceAllUsesWith(stats_op.arg());
    stats_op.erase();
  });
}

void PrepareDynamicRangeQuantizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  if (enable_float16_quantization_) {
    quant_specs_.inference_type = tensorflow::DT_HALF;
  }

  quant_specs_.disable_per_channel =
      !enable_dynamic_range_per_channel_quantization_;
  quant_specs_.minimum_elements_for_weights = min_elements_for_weights_;

  if (!enable_custom_op_quantization_.empty()) {
    ParseCustomOpSpecs(enable_custom_op_quantization_,
                       quant::CustomOpUpdateOptions::kINputIndices,
                       quant_specs_.custom_map);
  }

  ConvertTFLQuantOpsToMlirQuantOps(func);
  removeAllStatsOp(func);

  RewritePatternSet patterns(&getContext());
  patterns.add<PrepareDynamicRangeQuantizableOp>(ctx, quant_specs_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  ConvertMlirQuantOpsToTFLQuantOps(func);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect
// PrepareDynamicRangeQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>>
CreatePrepareDynamicRangeQuantizePass(
    const quant::QuantizationSpecs& quant_specs) {
  return std::make_unique<PrepareDynamicRangeQuantizePass>(quant_specs);
}

std::unique_ptr<OperationPass<func::FuncOp>>
CreatePrepareDynamicRangeQuantizePass() {
  return std::make_unique<PrepareDynamicRangeQuantizePass>();
}

}  // namespace TFL
}  // namespace mlir

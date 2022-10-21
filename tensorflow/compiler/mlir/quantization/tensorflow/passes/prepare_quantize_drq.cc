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
// Copied and modified from
// //third_party/tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc
// This transformation pass applies quantization propagation on TF dialect.

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize-drq Pass.
//
namespace mlir {
namespace quant {

namespace {

using QuantizationUnit = std::pair<Operation*, int>;
using QuantizationUnits = llvm::SetVector<QuantizationUnit>;

// Applies prepare quantization on the model in TF dialect for dynamic range
// quantization case.
class PrepareQuantizeDRQPass
    : public PassWrapper<PrepareQuantizeDRQPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, ::mlir::quant::QuantizationDialect,
                    ::mlir::quantfork::QuantizationForkDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizeDRQPass)

  // Constructor used by the PassRegistration and enforce int8 quantization.
  // This is only used by test.
  explicit PrepareQuantizeDRQPass() : op_set_(OpSet::UNIFORM_QUANTIZED) {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizeDRQPass(const QuantizationSpecs& quant_specs,
                                  OpSet op_set)
      : quant_specs_(quant_specs), op_set_(op_set) {
    enable_per_channel_quantization_ = !quant_specs_.disable_per_channel;
  }

  PrepareQuantizeDRQPass(const PrepareQuantizeDRQPass& other) {
    quant_specs_ = other.quant_specs_;
    op_set_ = other.op_set_;
    enable_per_channel_quantization_ = !quant_specs_.disable_per_channel;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-prepare-quantize-drq";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Prepare TF dialect for dynamic range quantization";
  }

  // The function might contain stats ops which are redundant for processing
  // dynamic range quantization. And stats ops may cause conflict while
  // processing the function for dynamic range quantization. Therefore, this
  // method preprocess the function to remove all stats ops.
  void removeAllStatsOp(func::FuncOp func);

  void runOnOperation() override;

 private:
  QuantizationSpecs quant_specs_;
  OpSet op_set_;

  Option<bool> enable_per_channel_quantization_{
      *this, "enable-per-channel-quantization", llvm::cl::init(false),
      llvm::cl::desc("Whether enable per-channel quantized weights.")};
};

// If the weight is applicable to dynamic range quantization, insert Quantize
// and Dequantize ops with per-tensor scale.
class PrepareDRQQuantizableOp : public OpRewritePattern<arith::ConstantOp> {
 public:
  explicit PrepareDRQQuantizableOp(MLIRContext* context,
                                   const quant::QuantizationSpecs& quant_specs,
                                   OpSet op_set,
                                   bool enable_per_channel_quantization)
      : OpRewritePattern<arith::ConstantOp>(context),
        quant_specs_(quant_specs),
        op_set_(op_set),
        enable_per_channel_quantization_(enable_per_channel_quantization) {}

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    QuantizationUnits quantizable_ops;

    // 1. Collect quantizable ops.
    if (!(getQuantizableOps(op, quantizable_ops))) {
      return failure();
    }

    // 2. Quantize collected ops. It is immediatly quantized by inserting Q-DQ
    // pair for int8.
    if (!(quantizeOps(rewriter, op, quantizable_ops))) {
      return failure();
    }

    return success();
  }

 private:
  // Mark users that are applicable for dynamic range quantization where the
  // criteria for determining quantizable ops differs by the inference type.
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
      std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(user);

      if (quant_specs_.inference_type == tensorflow::DT_QINT8 &&
          spec->quantizable_operands.contains(operand_num)) {
        quantizable_ops.insert({user, operand_num});
      }
    }
    return !quantizable_ops.empty();
  }

  // Check if transforming of the constant in the op is needed for the op_set.
  LogicalResult preprocessConstantOp(PatternRewriter& rewriter, OpSet op_set,
                                     arith::ConstantOp weight_op,
                                     QuantizationUnit quantizable_op) const {
    Operation* quantized_op = quantizable_op.first;
    int weight_operand_idx = quantizable_op.second;

    if (auto call_op = dyn_cast<TF::PartitionedCallOp>(quantized_op)) {
      const auto f_attr = call_op.fAttr().cast<FlatSymbolRefAttr>();
      // Non-quantizable op
      if (!call_op->hasAttr(kQuantTraitAttrName)) return failure();
      StringRef function_name = f_attr.getValue();
      if (!function_name.startswith("composite_")) {
        return failure();
      }
      if (function_name.contains("depthwise_conv2d")) {
        // Uniform Quantized op requires weights of tf.DepthwiseConv2dNative to
        // be transformed from [H,W,C,M] to [H,W,1,CxM] where
        // H=height,W=width,C=channel,M=multiplier. Instead of adding a reshape
        // op, directly modifying the weights as quantization occurs in the same
        // pass.
        if (op_set == OpSet::UNIFORM_QUANTIZED) {
          DenseFPElementsAttr attr;
          if (!matchPattern(weight_op->getResult(0), m_Constant(&attr))) {
            return failure();
          }

          // Get new shape
          auto cur_shape = attr.getType().getShape();
          TensorType new_shape = RankedTensorType::get(
              {cur_shape[0], cur_shape[1], 1, cur_shape[2] * cur_shape[3]},
              attr.getElementType());

          // Create a new const op with different shape
          std::vector<float> weight_values{
              attr.template getValues<float>().begin(),
              attr.template getValues<float>().end()};
          DenseFPElementsAttr new_attr = DenseFPElementsAttr::get(
              new_shape, llvm::makeArrayRef(weight_values));
          rewriter.setInsertionPointAfter(weight_op);
          auto new_weight_op = rewriter.create<arith::ConstantOp>(
              quantized_op->getLoc(), new_shape, new_attr);
          quantized_op->setOperand(weight_operand_idx, new_weight_op);

          // Fix function information accordingly
          auto module = call_op->getParentOfType<ModuleOp>();
          SymbolTable symbol_table(module);
          auto float_func =
              dyn_cast<func::FuncOp>(symbol_table.lookup(function_name));

          auto func_args = call_op.args();
          SmallVector<Value> new_func_args{func_args.begin(), func_args.end()};

          new_func_args[weight_operand_idx] = new_weight_op;
          float_func.getArgument(weight_operand_idx).setType(new_shape);
          float_func.setType(FunctionType::get(
              getContext(), TypeRange{ValueRange{new_func_args}},
              float_func.getResultTypes()));
        }
      }
    }
    return success();
  }

  // Apply per-tensor quantization for int8 dynamic range quantization.
  bool quantizeOpAsInt8(PatternRewriter& rewriter, arith::ConstantOp weight_op,
                        QuantizationUnit quantizable_op) const {
    Operation* quantized_op = quantizable_op.first;
    int weight_operand_idx = quantizable_op.second;

    std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(quantized_op);
    const bool is_narrow_range = true;
    const bool is_legacy_float = quant_specs_.legacy_float_scale;
    const bool is_signed = quant_specs_.IsSignedInferenceType();
    const int quant_dim = spec->coeff_op_quant_dim[weight_operand_idx];
    const bool is_per_channel_quantization =
        enable_per_channel_quantization_ && quant_dim != -1;
    const int bit_width = quant_specs_.GetQuantizationTypeWidth();

    if (failed(preprocessConstantOp(rewriter, op_set_, weight_op,
                                    quantizable_op))) {
      return false;
    }

    // Get attribute after preprocessing constant
    QuantizedType quant_type;
    DenseFPElementsAttr attr;
    auto preprocessed_weight_op = llvm::cast<arith::ConstantOp>(
        quantized_op->getOperand(weight_operand_idx).getDefiningOp());
    if (!matchPattern(preprocessed_weight_op->getResult(0),
                      m_Constant(&attr))) {
      return false;
    }

    if (is_per_channel_quantization) {
      quant_type = quant::GetUniformQuantizedPerAxisTypeForWeight(
                       attr, quant_dim,
                       /*symmetric=*/true, bit_width, is_signed,
                       is_narrow_range, is_legacy_float)
                       .template dyn_cast<quant::QuantizedType>();
    } else {
      quant_type = quant::GetUniformQuantizedTypeForWeight(
                       attr, is_narrow_range && is_signed, bit_width, is_signed,
                       is_narrow_range, is_legacy_float)
                       .template dyn_cast<quant::QuantizedType>();
    }
    return insertQDQ(rewriter, quantized_op, preprocessed_weight_op,
                     weight_operand_idx, quant_type);
  }

  // Insert Quantize and Dequantize ops.
  bool insertQDQ(PatternRewriter& rewriter, Operation* quantized_op,
                 arith::ConstantOp weight_op, int weight_operand_idx,
                 QuantizedType quant_type) const {
    if (!quant_type) return false;

    Type expressed_type = weight_op.getResult().getType();
    Type cast_type = quant_type.castFromExpressedType(expressed_type);

    // Insert DQ-op if it does not exist yet. Otherwise, just rewire without
    // creating a new DQ-op.
    for (auto connected_op : weight_op->getUsers()) {
      auto q_op =
          llvm::dyn_cast_or_null<quantfork::QuantizeCastOp>(connected_op);
      if (q_op && q_op.getType() == cast_type) {
        auto dq_op = llvm::cast<quantfork::DequantizeCastOp>(
            q_op.getResult().use_begin()->getOwner());
        quantized_op->setOperand(weight_operand_idx, dq_op);
        return false;
      }
    }
    rewriter.setInsertionPointAfter(weight_op);
    auto q = rewriter.create<quantfork::QuantizeCastOp>(
        weight_op->getLoc(), cast_type, weight_op.getResult());
    auto dq = rewriter.create<quantfork::DequantizeCastOp>(weight_op->getLoc(),
                                                           expressed_type, q);

    quantized_op->setOperand(weight_operand_idx, dq.getResult());
    return true;
  }

  // For each filtered user, apply quantization.
  bool quantizeOps(PatternRewriter& rewriter, arith::ConstantOp op,
                   QuantizationUnits& quantizable_ops) const {
    bool quantized = false;

    for (auto& quant_op : quantizable_ops) {
      if (quant_specs_.inference_type == tensorflow::DT_QINT8) {
        quantized |= quantizeOpAsInt8(rewriter, op, quant_op);
      }
    }
    return quantized;
  }

 protected:
  QuantizationSpecs quant_specs_;
  OpSet op_set_;
  bool enable_per_channel_quantization_;
};

// Remove all the stats ops which are redundant for dynamic range quantizaiton.
void PrepareQuantizeDRQPass::removeAllStatsOp(func::FuncOp func) {
  func.walk([&](quantfork::StatisticsOp stats_op) {
    stats_op.replaceAllUsesWith(stats_op.getArg());
    stats_op.erase();
  });
}

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_quantize.inc"

void PrepareQuantizeDRQPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  removeAllStatsOp(func);

  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.add<PrepareDRQQuantizableOp>(ctx, quant_specs_, op_set_,
                                        enable_per_channel_quantization_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareQuantizeDRQ
// pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizeDRQPass(
    const QuantizationSpecs& quant_specs, const OpSet op_set) {
  return std::make_unique<PrepareQuantizeDRQPass>(quant_specs, op_set);
}

static PassRegistration<PrepareQuantizeDRQPass> pass;

}  // namespace quant
}  // namespace mlir

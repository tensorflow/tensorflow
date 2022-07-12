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
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize-drq Pass.
//
namespace mlir {
namespace quant {

namespace {

using QuantizationUnits = llvm::SetVector<std::pair<Operation*, int>>;

// Applies prepare quantization on the model in TF dialect for dynamic range
// quantization case.
class PrepareQuantizeDRQPass
    : public PassWrapper<PrepareQuantizeDRQPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<TF::TensorFlowDialect, ::mlir::quant::QuantizationDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizeDRQPass)

  // Constructor used by the PassRegistration and enforce int8 quantization.
  // This is only used by test.
  explicit PrepareQuantizeDRQPass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizeDRQPass(const QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {}

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
};

// If the weight is applicable to dynamic range quantization, insert Quantize
// and Dequantize ops with per-tensor scale.
class PrepareDRQQuantizableOp : public OpRewritePattern<arith::ConstantOp> {
 public:
  explicit PrepareDRQQuantizableOp(MLIRContext* context,
                                   const quant::QuantizationSpecs& quant_specs)
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
    // pair for int8.
    if (!(quantizeOps(rewriter, op, quantizable_ops))) {
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
  // quantization.
  bool hasInt8QuantizableOperandAt(Operation* op, int operand_index) const {
    if (auto call_op = dyn_cast<TF::PartitionedCallOp>(op)) {
      StringRef function_name =
          call_op.fAttr().cast<FlatSymbolRefAttr>().getValue();
      if (!function_name.startswith("composite_")) {
        return false;
      }
      if (function_name.contains("matmul")) {
        return isQuantizableIndex(operand_index, {1});
      }
    }
    return false;
  }

  // Apply per-tensor quantization for int8 dynamic range quantization.
  bool quantizeOpAsInt8(PatternRewriter& rewriter, arith::ConstantOp op,
                        std::pair<Operation*, int> quant_op) const {
    bool is_narrow_range = true;
    bool is_legacy_float = quant_specs_.legacy_float_scale;
    bool is_signed = quant_specs_.IsSignedInferenceType();
    int bit_width = quant_specs_.GetQuantizationTypeWidth();

    QuantizedType quant_type = nullptr;
    DenseFPElementsAttr attr;
    if (!matchPattern(op->getResult(0), m_Constant(&attr))) return false;

    quant_type = quant::GetUniformQuantizedTypeForWeight(
                     attr, is_narrow_range && is_signed, bit_width, is_signed,
                     is_narrow_range, is_legacy_float)
                     .template dyn_cast<quant::QuantizedType>();

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
      auto q_op = llvm::dyn_cast_or_null<quant::QuantizeCastOp>(connected_op);
      if (q_op && q_op.getType() == cast_type) {
        auto dq_op = llvm::cast<quant::DequantizeCastOp>(
            q_op.getResult().use_begin()->getOwner());
        quantize_op->setOperand(quantize_operand_num, dq_op);
        return false;
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<quant::QuantizeCastOp>(op->getLoc(), cast_type,
                                                    op.getResult());
    auto dq = rewriter.create<quant::DequantizeCastOp>(op->getLoc(),
                                                       expressed_type, q);
    quantize_op->setOperand(quantize_operand_num, dq.getResult());
    return true;
  }

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

      if (quant_specs_.inference_type == tensorflow::DT_QINT8 &&
          hasInt8QuantizableOperandAt(user, operand_num)) {
        quantizable_ops.insert({user, operand_num});
      }
    }
    return !quantizable_ops.empty();
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
  quant::QuantizationSpecs quant_specs_;
};

// Remove all the stats ops which are redundant for dynamic range quantizaiton.
void PrepareQuantizeDRQPass::removeAllStatsOp(func::FuncOp func) {
  func.walk([&](quant::StatisticsOp stats_op) {
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
  patterns.add<PrepareDRQQuantizableOp>(ctx, quant_specs_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareQuantizeDRQ
// pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizeDRQPass() {
  return std::make_unique<PrepareQuantizeDRQPass>();
}

static PassRegistration<PrepareQuantizeDRQPass> pass;

}  // namespace quant
}  // namespace mlir

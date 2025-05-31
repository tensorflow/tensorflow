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

#include <memory>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/temp_tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize-drq Pass.
//
namespace mlir {
namespace quant {

namespace {

using QuantizationUnit = std::pair<Operation*, int>;
using QuantizationUnits = llvm::SetVector<QuantizationUnit>;
using ::mlir::tf_quant::GetTFOpQuantSpec;
using ::mlir::tf_quant::GetUniformQuantizedPerAxisTypeForWeight;
using ::mlir::tf_quant::GetUniformQuantizedTypeForWeight;
using ::mlir::tf_quant::OpQuantSpec;
using ::mlir::tf_quant::QuantizationSpecs;
using ::tensorflow::quantization::OpSet;

// Applies prepare quantization on the model in TF dialect for dynamic range
// quantization case.
class PrepareQuantizeDRQPass
    : public PassWrapper<PrepareQuantizeDRQPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, ::mlir::quant::QuantDialect,
                    ::mlir::quant::ir::TFQuantDialect>();
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

    // 2. Quantize collected ops. It is immediately quantized by inserting Q-DQ
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
    auto type = mlir::dyn_cast<ShapedType>(op.getType());
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

  // Apply per-tensor quantization for int8 dynamic range quantization.
  bool quantizeOpAsInt8(PatternRewriter& rewriter, arith::ConstantOp op,
                        QuantizationUnit quant_op) const {
    auto [quantized_op, weight_idx] = quant_op;
    const bool is_narrow_range = true;
    const bool is_legacy_float = quant_specs_.legacy_float_scale;
    const bool is_signed = quant_specs_.IsSignedInferenceType();
    const int bit_width = quant_specs_.GetQuantizationTypeWidth();

    std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(quantized_op);
    const int quant_dim = spec->coeff_op_quant_dim[weight_idx];
    const bool is_per_channel_quantization =
        enable_per_channel_quantization_ && quant_dim != -1;

    QuantizedType quant_type;
    DenseFPElementsAttr attr;
    if (!matchPattern(op->getResult(0), m_Constant(&attr))) return false;

    if (attr.size() < quant_specs_.minimum_elements_for_weights) {
      op->emitRemark("Quantization is skipped for ")
          << quantized_op->getName().getStringRef().str() << " because it has "
          << mlir::dyn_cast<DenseFPElementsAttr>(attr).size()
          << " elements which is fewer than the threshold("
          << quant_specs_.minimum_elements_for_weights << " elements).";
      return false;
    }

    if (is_per_channel_quantization) {
      quant_type = mlir::dyn_cast<quant::QuantizedType>(
          quant::GetUniformQuantizedPerAxisTypeForWeight(
              attr, quant_dim,
              /*symmetric=*/true, bit_width, is_signed, is_narrow_range,
              is_legacy_float));
    } else {
      quant_type = mlir::dyn_cast<quant::QuantizedType>(
          quant::GetUniformQuantizedTypeForWeight(
              attr, is_narrow_range && is_signed, bit_width, is_signed,
              is_narrow_range, is_legacy_float));
    }
    return insertQDQ(rewriter, op, quant_type, quant_op);
  }

  // Insert Quantize and Dequantize ops.
  bool insertQDQ(PatternRewriter& rewriter, arith::ConstantOp op,
                 QuantizedType quant_type, QuantizationUnit quant_op) const {
    if (!quant_type) return false;

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    Type expressed_type = op.getResult().getType();
    Type cast_type = quant_type.castFromExpressedType(expressed_type);

    // Insert DQ-op if it does not exist yet. Otherwise, just rewire without
    // creating a new DQ-op.
    for (auto connected_op : op->getUsers()) {
      auto q_op =
          llvm::dyn_cast_or_null<mlir::quant::ir::QuantizeCastOp>(connected_op);
      if (q_op && q_op.getType() == cast_type) {
        auto dq_op = llvm::cast<mlir::quant::ir::DequantizeCastOp>(
            q_op.getResult().use_begin()->getOwner());
        quantize_op->setOperand(quantize_operand_num, dq_op);
        return false;
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<mlir::quant::ir::QuantizeCastOp>(
        op->getLoc(), cast_type, op.getResult());
    auto dq = rewriter.create<mlir::quant::ir::DequantizeCastOp>(
        op->getLoc(), expressed_type, q);
    quantize_op->setOperand(quantize_operand_num, dq.getResult());
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
  func.walk([&](mlir::quant::ir::StatisticsOp stats_op) {
    stats_op.replaceAllUsesWith(stats_op.getArg());
    stats_op.erase();
  });
}

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_quantize.inc"

void PrepareQuantizeDRQPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module_op = getOperation();

  populateWithGenerated(patterns);
  patterns.add<PrepareDRQQuantizableOp>(ctx, quant_specs_, op_set_,
                                        enable_per_channel_quantization_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  for (auto func : module_op.getOps<func::FuncOp>()) {
    removeAllStatsOp(func);
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-prepare-quantize-drq failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareQuantizeDRQ
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareQuantizeDRQPass(
    const QuantizationSpecs& quant_specs, const OpSet op_set) {
  return std::make_unique<PrepareQuantizeDRQPass>(quant_specs, op_set);
}

static PassRegistration<PrepareQuantizeDRQPass> pass;

}  // namespace quant
}  // namespace mlir

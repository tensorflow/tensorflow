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

// NOLINTNEXTLINE
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> dynamic_quantize_disable_per_channel(
    "tfl-dynamic-quantize-disable-per-channel", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether disable per-channel quantized weights."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// The prepare-dynamic-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {

using QuantizationUnits = llvm::SetVector<std::pair<Operation*, int>>;

// Applies prepare dynamic quantization on the model in TFL dialect.
// This pass runs before the quantization pass and apply preprocess if
// applicable.
class PrepareDynamicQuantizePass
    : public PassWrapper<PrepareDynamicQuantizePass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<TensorFlowLiteDialect, ::mlir::quant::QuantizationDialect>();
  }

 public:
  // Constructor used by the PassRegistration and enforce int8 quantization.
  // This is only used by test.
  explicit PrepareDynamicQuantizePass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
    quant_specs_.weight_quantization = true;
    quant_specs_.enable_mlir_dynamic_quantizer = true;
    quant_specs_.disable_per_channel = dynamic_quantize_disable_per_channel;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareDynamicQuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {}

  StringRef getArgument() const final { return "tfl-prepare-dynamic-quantize"; }
  StringRef getDescription() const final {
    return "Prepare TFL dialect for dynamic quantization";
  }

  void runOnFunction() override;

 private:
  QuantizationSpecs quant_specs_;
};

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

// If the weight is applicable to dynamic quantization, insert Quantize and
// Dequantize ops with either per-axis or per-tensor scale.
class PreprocessDynamicQuantizableOp
    : public OpRewritePattern<arith::ConstantOp> {
 public:
  explicit PreprocessDynamicQuantizableOp(MLIRContext* context,
                                          const QuantizationSpecs& quant_specs)
      : OpRewritePattern<arith::ConstantOp>(context),
        quant_specs_(quant_specs) {}

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    QuantizationUnits quantizable_ops;

    if (!(getQuantizableOps(op, quantizable_ops))) {
      return failure();
    }

    if (!(quantizeOps(op, quantizable_ops, rewriter))) {
      return failure();
    }
    return success();
  }

 private:
  // Mark users that are applicable for dynamic quantization if it
  // uses float tensors which are not biases and is a DynamicQuantizableOp.
  bool getQuantizableOps(arith::ConstantOp op,
                         QuantizationUnits& quantizable_ops) const {
    // Non-float tensors do not need quantization.
    auto type = op.getType().dyn_cast<ShapedType>();
    if (!type || !type.getElementType().isa<FloatType>()) return false;

    Value value = op.getResult();

    // Check whether dynamic-quantization can be applied.
    // TODO(b/201599094): check whether weight size < 1024 condition is needed
    // here
    for (auto& use : value.getUses()) {
      Operation* user = use.getOwner();
      int operand_num = use.getOperandNumber();

      auto spec = GetOpQuantSpec(user);
      auto biases = spec->biases_params;

      if (biases.find(operand_num) == biases.end() &&
          user->hasTrait<OpTrait::quant::DynamicQuantizableOp>()) {
        quantizable_ops.insert({user, operand_num});
      }
    }
    return !quantizable_ops.empty();
  }

  // For each filtered user, apply quantization.
  bool quantizeOps(arith::ConstantOp op, QuantizationUnits& quantizable_ops,
                   PatternRewriter& rewriter) const {
    bool quantized = false;
    for (auto& quant_op : quantizable_ops) {
      quantized |= quantizeOp(op, quant_op, rewriter);
    }
    return quantized;
  }

  // Apply per-axis quantization if applicable. Otherwise, apply per-tensor
  // quantization.
  bool quantizeOp(arith::ConstantOp op, std::pair<Operation*, int> quant_op,
                  PatternRewriter& rewriter) const {
    bool is_signed = quant_specs_.IsSignedInferenceType();
    int bit_width = quant_specs_.GetQuantizationTypeWidth();

    Operation* quantize_op = quant_op.first;
    int quantize_operand_num = quant_op.second;

    auto affine_user =
        llvm::dyn_cast<mlir::AffineQuantizedOpInterface>(quantize_op);

    bool op_with_narrow_range =
        affine_user &&
        affine_user.GetAffineOperandIndex() == quantize_operand_num &&
        affine_user.RequiredNarrowRangeAffineOperand();

    bool op_with_per_axis_support =
        op_with_narrow_range && affine_user.GetQuantizationDimIndex() != -1 &&
        !quant_specs_.disable_per_channel;

    QuantizedType quant_type = nullptr;
    DenseFPElementsAttr attr;
    if (!matchPattern(op->getResult(0), m_Constant(&attr))) return false;

    if (op_with_per_axis_support) {
      quant_type = quant::GetUniformQuantizedPerAxisTypeForWeight(
                       attr, affine_user.GetQuantizationDimIndex(),
                       /*symmetric=*/true, bit_width, is_signed,
                       /*narrow_range=*/true, quant_specs_.legacy_float_scale)
                       .template dyn_cast<quant::QuantizedType>();
    } else {
      quant_type =
          quant::GetUniformQuantizedTypeForWeight(
              attr, op_with_narrow_range && is_signed, bit_width, is_signed,
              op_with_narrow_range, quant_specs_.legacy_float_scale)
              .template dyn_cast<quant::QuantizedType>();
    }
    return insertQDQ(op, quantize_op, quant_type, quantize_operand_num,
                     rewriter);
  }

  // Insert Quantize and Dequantize ops.
  bool insertQDQ(arith::ConstantOp op, Operation* quantize_op,
                 QuantizedType quant_type, int quantize_operand_num,
                 PatternRewriter& rewriter) const {
    if (!quant_type) return false;

    Type expressed_type = op->getResult(0).getType();
    Type cast_type = quant_type.castFromExpressedType(expressed_type);
    rewriter.setInsertionPointAfter(op);
    auto q = rewriter.create<Q>(op->getLoc(), cast_type, op->getResult(0));
    auto dq = rewriter.create<DQ>(op->getLoc(), expressed_type, q);
    quantize_op->setOperand(quantize_operand_num, dq.getResult());
    return true;
  }

 protected:
  QuantizationSpecs quant_specs_;
};

void PrepareDynamicQuantizePass::runOnFunction() {
  FuncOp func = getFunction();
  MLIRContext* ctx = func.getContext();

  ConvertTFLQuantOpsToMlirQuantOps(func);

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<PreprocessDynamicQuantizableOp>(ctx, quant_specs_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  ConvertMlirQuantOpsToTFLQuantOps(func);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareDynamicQuantize
// pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareDynamicQuantizePass(
    const QuantizationSpecs& quant_specs) {
  return std::make_unique<PrepareDynamicQuantizePass>(quant_specs);
}

static PassRegistration<PrepareDynamicQuantizePass> pass;

}  // namespace TFL
}  // namespace mlir

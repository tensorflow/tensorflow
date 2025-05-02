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
// This transformation pass applies quantization propagation on TF dialect.

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// The preprocess-op Pass.
//
namespace mlir {
namespace tf_quant {

namespace {

using QuantMethod =
    ::tensorflow::quantization::QuantizationMethod::PresetMethod;
using QuantizationUnit = std::pair<Operation*, int>;
using QuantizationUnits = llvm::SetVector<QuantizationUnit>;
using ::tensorflow::quantization::OpSet;

// Preprocesses ops to allow multi-axis quantization, prior to quantization
// passes. Currently, per-channel quantization only supports 1D results.
class TFPreprocessOpPass
    : public PassWrapper<TFPreprocessOpPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, quant::QuantDialect,
                    mlir::quant::ir::TFQuantDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFPreprocessOpPass)

  explicit TFPreprocessOpPass() = default;

  // Constructor used by manually creating the pass.
  explicit TFPreprocessOpPass(OpSet op_set,
                      const QuantMethod quantization_method,
                      bool enable_per_channel_quantization) {
    op_set_ = op_set;
    quantization_method_ = quantization_method;
    enable_per_channel_quantization_ = enable_per_channel_quantization;
  }

  TFPreprocessOpPass(const TFPreprocessOpPass& other) {
    op_set_ = other.op_set_;
    quantization_method_ = other.quantization_method_;
    enable_per_channel_quantization_ = other.enable_per_channel_quantization_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-preprocess-op";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Preprocess TF op prior to quantization";
  }

  void runOnOperation() override;

 private:
  Option<OpSet> op_set_{
      *this, "target-opset", llvm::cl::init(OpSet::UNIFORM_QUANTIZED),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};

  Option<QuantMethod> quantization_method_{
      *this, "quantization-method",
      llvm::cl::init(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8),
      llvm::cl::desc("Choose quantization method."),
      llvm::cl::values(
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8,
                     "ptq", "Post-training static-range quantization"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_DYNAMIC_RANGE_INT8,
                     "drq", "Post-training dynamic-range quantizaiton"),
          clEnumValN(tensorflow::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8,
                     "weight_only", "Post-training weight-only quantizaiton"))};

  Option<bool> enable_per_channel_quantization_{
      *this, "enable-per-channel-quantization", llvm::cl::init(false),
      llvm::cl::desc("Whether enable per-channel quantized weights.")};
};

// Apply constant transformations for the op_set.
class PreprocessConstantOp : public OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit PreprocessConstantOp(MLIRContext* context, OpSet op_set,
                                QuantMethod quantization_method,
                                bool enable_per_channel_quantization)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        op_set_(op_set),
        quantization_method_(quantization_method),
        enable_per_channel_quantization_(enable_per_channel_quantization) {}

  LogicalResult addReshapeOpToDepthwiseWeight(TF::PartitionedCallOp op,
                                              PatternRewriter& rewriter,
                                              StringRef function_name) const {
    std::unique_ptr<quant::OpQuantSpec> spec = quant::GetTFOpQuantSpec(op);
    const absl::flat_hash_set<int> operands = spec->quantizable_operands;

    if (operands.size() != 1) return failure();
    int weight_operand_idx = *operands.begin();

    Operation* weight_op = op.getOperand(weight_operand_idx).getDefiningOp();
    DenseFPElementsAttr attr;
    if (!matchPattern(weight_op->getResult(0), m_Constant(&attr))) {
      return failure();
    }

    // Get new shape.
    llvm::ArrayRef<int64_t> cur_shape = attr.getType().getShape();
    int cur_rank = cur_shape.size();
    if (cur_rank != 4 || cur_shape[2] == 1) return failure();
    TensorType new_shape = RankedTensorType::get(
        {cur_shape[0], cur_shape[1], 1, cur_shape[2] * cur_shape[3]},
        attr.getElementType());

    // Inserts a reshape op.
    auto shape_spec_type =
        RankedTensorType::get({cur_rank}, rewriter.getIntegerType(64));
    auto new_shape_const_attr =
        DenseElementsAttr::get(shape_spec_type, new_shape.getShape());
    rewriter.setInsertionPointAfter(weight_op);
    auto new_shape_const = rewriter.create<arith::ConstantOp>(
        weight_op->getLoc(), shape_spec_type, new_shape_const_attr);
    auto reshape_op = rewriter.create<TF::ReshapeOp>(
        weight_op->getLoc(), new_shape, weight_op->getResult(0),
        new_shape_const);
    op->setOperand(weight_operand_idx, reshape_op);

    // Create a new function with preprocessed types.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);
    func::FuncOp float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(function_name));
    OperandRange func_args = op.getArgs();
    func::FuncOp new_float_func = float_func.clone();

    SmallVector<Value> new_float_func_args{func_args.begin(), func_args.end()};
    new_float_func_args[weight_operand_idx] = reshape_op;
    new_float_func.getArgument(weight_operand_idx).setType(new_shape);
    new_float_func.setType(FunctionType::get(
        getContext(), TypeRange{ValueRange{new_float_func_args}},
        new_float_func.getResultTypes()));
    symbol_table.insert(new_float_func);

    op->setAttr("f", SymbolRefAttr::get(rewriter.getContext(),
                                        new_float_func.getName()));

    return success();
  }

  LogicalResult matchAndRewrite(TF::PartitionedCallOp op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = mlir::dyn_cast<FlatSymbolRefAttr>(op.getFAttr());
    // Non-quantizable op
    if (!op->hasAttr(quant::kQuantTraitAttrName)) return failure();
    StringRef function_name = f_attr.getValue();
    // TODO(b/228928859): Improve the getter function to match attributes rather
    // than function name.
    if (!function_name.starts_with("composite_")) {
      return failure();
    }

    if (function_name.contains("depthwise_conv2d")) {
      // Uniform Quantized op requires weights of tf.DepthwiseConv2dNative to
      // be transformed from [H,W,C,M] to [H,W,1,CxM] where
      // H=height,W=width,C=channel,M=multiplier. Therefore, a reshape op is
      // inserted between the constant op and the function op so that the
      // constant is safely transformed for the multi-use cases as well. Note
      // that bias doesn't need transformation as its shape is already in [CxM].
      if (op_set_ == OpSet::UNIFORM_QUANTIZED ||
          (op_set_ == OpSet::XLA && enable_per_channel_quantization_ &&
           quantization_method_ ==
               tensorflow::quantization::QuantizationMethod::
                   METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8)) {
        return addReshapeOpToDepthwiseWeight(op, rewriter, function_name);
      }
    }
    return failure();
  }

 private:
  const OpSet op_set_;
  const QuantMethod quantization_method_;
  const bool enable_per_channel_quantization_;
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/preprocess_op.inc"

void TFPreprocessOpPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module_op = getOperation();

  populateWithGenerated(patterns);
  patterns.add<PreprocessConstantOp>(ctx, op_set_, quantization_method_,
                                     enable_per_channel_quantization_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-preprocess-op failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PreprocessOp
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFPreprocessOpPass(
    const OpSet op_set, QuantMethod quantization_method,
    const bool enable_per_channel_quantization) {
  return std::make_unique<TFPreprocessOpPass>(op_set, quantization_method,
                                            enable_per_channel_quantization);
}

static PassRegistration<TFPreprocessOpPass> pass;

}  // namespace tf_quant
}  // namespace mlir

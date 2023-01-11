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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

//===----------------------------------------------------------------------===//
// The preprocess-op Pass.
//
namespace mlir {
namespace quant {

namespace {

using QuantizationUnit = std::pair<Operation*, int>;
using QuantizationUnits = llvm::SetVector<QuantizationUnit>;

// Preprocesses ops to allow multi-axis quantization, prior to quantization
// passes. Currently, per-channel quantization only supports 1D results.
class PreprocessOpPass
    : public PassWrapper<PreprocessOpPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, QuantizationDialect,
                    quantfork::QuantizationForkDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreprocessOpPass)

  // Constructor used by the PassRegistration and enforce int8 quantization.
  // This is only used by test.
  explicit PreprocessOpPass() : op_set_(OpSet::UNIFORM_QUANTIZED) {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
  }

  // Constructor used by manually creating the pass.
  explicit PreprocessOpPass(const QuantizationSpecs& quant_specs, OpSet op_set)
      : quant_specs_(quant_specs), op_set_(op_set) {}

  PreprocessOpPass(const PreprocessOpPass& other) {
    quant_specs_ = other.quant_specs_;
    op_set_ = other.op_set_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-preprocess-op";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Preprocess TF op prior to quantization";
  }

  void runOnOperation() override;

 private:
  QuantizationSpecs quant_specs_;
  OpSet op_set_;
};

// Apply constant transformations for the op_set.
class PreprocessConstantOp : public OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit PreprocessConstantOp(MLIRContext* context, OpSet op_set)
      : OpRewritePattern<TF::PartitionedCallOp>(context), op_set_(op_set) {}

  LogicalResult matchAndRewrite(TF::PartitionedCallOp op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
    // Non-quantizable op
    if (!op->hasAttr(kQuantTraitAttrName)) return failure();
    StringRef function_name = f_attr.getValue();
    if (!function_name.startswith("composite_")) {
      return failure();
    }

    std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(op);
    const absl::flat_hash_set<int> operands = spec->quantizable_operands;

    if (function_name.contains("depthwise_conv2d")) {
      // Uniform Quantized op requires weights of tf.DepthwiseConv2dNative to
      // be transformed from [H,W,C,M] to [H,W,1,CxM] where
      // H=height,W=width,C=channel,M=multiplier. Therefore, a reshape op is
      // inserted between the constant op and the function op so that the
      // constant is safely transformed for the multi-use cases as well. Note
      // that bias doesn't need transformation as its shape is already in [CxM].
      if (operands.size() != 1) return failure();
      int weight_operand_idx = *operands.begin();
      Operation* weight_op = op.getOperand(weight_operand_idx).getDefiningOp();

      if (op_set_ == OpSet::UNIFORM_QUANTIZED) {
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

        SmallVector<Value> new_float_func_args{func_args.begin(),
                                               func_args.end()};
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
    }
    return failure();
  }

 private:
  const OpSet op_set_;
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/preprocess_op.inc"

void PreprocessOpPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module_op = getOperation();

  populateWithGenerated(patterns);
  patterns.add<PreprocessConstantOp>(ctx, op_set_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-preprocess-op failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PreprocessOp
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePreprocessOpPass(
    const QuantizationSpecs& quant_specs, const OpSet op_set) {
  return std::make_unique<PreprocessOpPass>(quant_specs, op_set);
}

static PassRegistration<PreprocessOpPass> pass;

}  // namespace quant
}  // namespace mlir

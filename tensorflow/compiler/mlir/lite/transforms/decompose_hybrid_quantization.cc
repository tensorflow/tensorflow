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

// This transformation pass decomposes dense operations that assume
// support for hybrid quantization. These cases cover when a dense operation
// (e.g. matmul) has both quantized and unquantized inputs by dequantizing
// the quantized inputs, performing the operation in the expressed type, then
// requantizing if a quantized output is required.
//
// The motivation behind these changes is for Dialects that assume only float
// or quantized computation, and do not support a mixture of these types on
// dense operations. Decomposition allows TFLite to be compiled to these
// dialects, such as TOSA.

#include <utility>

#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {

namespace {

#define GEN_PASS_DEF_DECOMPOSEHYBRIDQUANTIZATIONPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

class DecomposeHybridQuantizationPass
    : public impl::DecomposeHybridQuantizationPassBase<
          DecomposeHybridQuantizationPass> {
 public:
  explicit DecomposeHybridQuantizationPass() {}
  void runOnOperation() override;
};

template <typename SrcOp>
class DequantizeConverter : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp srcop,
                                PatternRewriter &rewriter) const final {
    Operation *op = srcop.getOperation();
    bool allTypesFp = true;
    bool allTypesQuantizedOrInt = true;
    for (auto operand : op->getOperands()) {
      ShapedType type = operand.getType().template dyn_cast<ShapedType>();
      if (!type) continue;
      allTypesFp &= !type.getElementType().isa<quant::QuantizedType>();
      allTypesQuantizedOrInt &=
          (type.getElementType().isa<quant::QuantizedType>() ||
           type.getElementType().isa<IntegerType>());
    }

    for (auto result : op->getResults()) {
      ShapedType type = result.getType().template cast<ShapedType>();
      allTypesFp &= !type.getElementType().isa<quant::QuantizedType>();
      allTypesQuantizedOrInt &=
          (type.getElementType().isa<quant::QuantizedType>() ||
           type.getElementType().isa<IntegerType>());
    }

    // If all quantized or floating point then types are consistent.
    // Int is valid in combination with both quantized and floating point.
    // This occurs when doing qi16 convolution, as bias is passed as a
    // non-quantized int64
    if (allTypesFp || allTypesQuantizedOrInt) return failure();

    Location loc = op->getLoc();
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (auto operand : op->getOperands()) {
      if (QuantizedType::getQuantizedElementType(operand.getType())) {
        auto newTy = QuantizedType::castToExpressedType(operand.getType());
        newOperands.push_back(
            rewriter.create<TFL::DequantizeOp>(loc, newTy, operand));
        continue;
      }

      newOperands.push_back(operand);
    }

    SmallVector<Type> newResultTys;
    for (auto result : op->getResults()) {
      Type resultTy = result.getType();
      if (QuantizedType::getQuantizedElementType(resultTy)) {
        resultTy = QuantizedType::castToExpressedType(resultTy);
      }
      newResultTys.push_back(resultTy);
    }

    auto newResults = rewriter
                          .create<SrcOp>(loc, newResultTys, newOperands,
                                         op->getAttrDictionary().getValue())
                          .getOperation()
                          ->getResults();

    SmallVector<Value> replaceResults;
    for (int i = 0; i < newResults.size(); i++) {
      Value result = newResults[i];
      Type resultTy = op->getOpResult(i).getType();
      if (QuantizedType::getQuantizedElementType(resultTy)) {
        replaceResults.push_back(rewriter.create<TFL::QuantizeOp>(
            loc, resultTy, result, TypeAttr::get(resultTy)));
        continue;
      }

      replaceResults.push_back(result);
    }

    rewriter.replaceOp(op, replaceResults);

    return success();
  }
};

void DecomposeHybridQuantizationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto *ctx = &getContext();
  auto func = getOperation();
  patterns.add<DequantizeConverter<TFL::Conv2DOp>,
               DequantizeConverter<TFL::Conv3DOp>,
               DequantizeConverter<TFL::DepthwiseConv2DOp>,
               DequantizeConverter<TFL::FullyConnectedOp>,
               DequantizeConverter<TFL::TransposeConvOp>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDecomposeHybridQuantizationPass() {
  return std::make_unique<DecomposeHybridQuantizationPass>();
}

}  // namespace TFL
}  // namespace mlir

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

#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/Passes.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantizeUtils.h"

namespace mlir {
namespace quant::ir {

using mlir::quant::QuantizedType;

namespace {
#define GEN_PASS_DEF_QUANTCONVERTCONST
#include "tensorflow/compiler/mlir/quantization/common/ir/Passes.h.inc"

struct ConvertConstPass : public impl::QuantConvertConstBase<ConvertConstPass> {
  void runOnOperation() override;
};

struct QuantizedConstRewrite : public OpRewritePattern<QuantizeCastOp> {
  using OpRewritePattern<QuantizeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeCastOp qbarrier,
                                PatternRewriter &rewriter) const override;
};

}  // namespace

/// Matches a [constant] -> [qbarrier] where the qbarrier results type is
/// quantized and the operand type is quantizable.

LogicalResult QuantizedConstRewrite::matchAndRewrite(
    QuantizeCastOp qbarrier, PatternRewriter &rewriter) const {
  Attribute value;

  // Is the operand a constant?
  if (!matchPattern(qbarrier.getArg(), m_Constant(&value))) {
    return failure();
  }

  // Does the qbarrier convert to a quantized type. This will not be true
  // if a quantized type has not yet been chosen or if the cast to an equivalent
  // storage type is not supported.
  Type qbarrierResultType = qbarrier.getResult().getType();
  QuantizedType quantizedElementType =
      QuantizedType::getQuantizedElementType(qbarrierResultType);
  if (!quantizedElementType) {
    return failure();
  }
  if (!QuantizedType::castToStorageType(qbarrierResultType)) {
    return failure();
  }

  // Is the operand type compatible with the expressed type of the quantized
  // type? This will not be true if the qbarrier is superfluous (converts
  // from and to a quantized type).
  if (!quantizedElementType.isCompatibleExpressedType(
          qbarrier.getArg().getType())) {
    return failure();
  }

  // Is the constant value a type expressed in a way that we support?
  if (!mlir::isa<FloatAttr, DenseElementsAttr, SparseElementsAttr>(value)) {
    return failure();
  }

  Type newConstValueType;
  auto newConstValue =
      quantizeAttr(value, quantizedElementType, newConstValueType);
  if (!newConstValue) {
    return failure();
  }

  // When creating the new const op, use a fused location that combines the
  // original const and the qbarrier that led to the quantization.
  auto fusedLoc = rewriter.getFusedLoc(
      {qbarrier.getArg().getDefiningOp()->getLoc(), qbarrier.getLoc()});
  auto newConstOp = rewriter.create<arith::ConstantOp>(
      fusedLoc, newConstValueType, cast<TypedAttr>(newConstValue));
  rewriter.replaceOpWithNewOp<StorageCastOp>(qbarrier, qbarrier.getType(),
                                             newConstOp);
  return success();
}

void ConvertConstPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto *context = &getContext();
  patterns.add<QuantizedConstRewrite>(context);
  (void)applyPatternsGreedily(func, std::move(patterns));
}

std::unique_ptr<OperationPass<func::FuncOp>> createConvertConstPass() {
  return std::make_unique<ConvertConstPass>();
}

}  // namespace quant::ir
}  // namespace mlir

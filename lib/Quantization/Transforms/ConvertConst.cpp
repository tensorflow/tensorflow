//===- ConvertConst.cpp - Quantizes constant ops --------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Quantization/Passes.h"
#include "mlir/Quantization/QuantOps.h"
#include "mlir/Quantization/QuantizeUtils.h"
#include "mlir/Quantization/UniformSupport.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;
using namespace mlir::quant;

namespace {

class ConvertConstPass : public FunctionPass<ConvertConstPass> {
public:
  void runOnFunction() override;
};

class QuantizedConstRewrite : public RewritePattern {
public:
  struct State : PatternState {
    QuantizedType quantizedElementType;
    Attribute value;
  };

  QuantizedConstRewrite(MLIRContext *context)
      : RewritePattern(QuantizeCastOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Operation *op) const override;
  void rewrite(Operation *op, std::unique_ptr<PatternState> baseState,
               PatternRewriter &rewriter) const override;
};

} // end anonymous namespace

/// Matches a [constant] -> [qbarrier] where the qbarrier results type is
/// quantized and the operand type is quantizable.
PatternMatchResult QuantizedConstRewrite::match(Operation *op) const {
  State state;

  // Is the operand a constant?
  auto qbarrier = cast<QuantizeCastOp>(op);
  if (!matchPattern(qbarrier.arg(), m_Constant(&state.value))) {
    return matchFailure();
  }
  // Does the qbarrier convert to a quantized type. This will not be true
  // if a quantized type has not yet been chosen or if the cast to an equivalent
  // storage type is not supported.
  Type qbarrierResultType = qbarrier.getResult()->getType();
  state.quantizedElementType =
      QuantizedType::getQuantizedElementType(qbarrierResultType);
  if (!state.quantizedElementType) {
    return matchFailure();
  }
  if (!QuantizedType::castToStorageType(qbarrierResultType)) {
    return matchFailure();
  }

  // Is the operand type compatible with the expressed type of the quantized
  // type? This will not be true if the qbarrier is superfluous (converts
  // from and to a quantized type).
  if (!state.quantizedElementType.isCompatibleExpressedType(
          qbarrier.arg()->getType())) {
    return matchFailure();
  }

  // Is the constant value a type expressed in a way that we support?
  if (!state.value.isa<FloatAttr>() && !state.value.isa<SplatElementsAttr>() &&
      !state.value.isa<DenseElementsAttr>() &&
      !state.value.isa<SparseElementsAttr>()) {
    return matchFailure();
  }

  return matchSuccess(llvm::make_unique<State>(std::move(state)));
}

void QuantizedConstRewrite::rewrite(Operation *op,
                                    std::unique_ptr<PatternState> baseState,
                                    PatternRewriter &rewriter) const {
  auto state = static_cast<State *>(baseState.get());

  Type newConstValueType;
  Attribute newConstValue = quantizeAttr(
      state->value, state->quantizedElementType, newConstValueType);
  if (!newConstValue) {
    return;
  }

  auto *origConstOp = op->getOperand(0);
  // When creating the new const op, use a fused location that combines the
  // original const and the qbarrier that led to the quantization.
  auto fusedLoc =
      FusedLoc::get({origConstOp->getDefiningOp()->getLoc(), op->getLoc()},
                    rewriter.getContext());
  auto newConstOp =
      rewriter.create<ConstantOp>(fusedLoc, newConstValueType, newConstValue);
  rewriter.replaceOpWithNewOp<StorageCastOp>(
      op, {origConstOp}, *op->result_type_begin(), newConstOp);
}

void ConvertConstPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto &func = getFunction();
  auto *context = &getContext();
  patterns.push_back(llvm::make_unique<QuantizedConstRewrite>(context));
  applyPatternsGreedily(func, std::move(patterns));
}

FunctionPassBase *mlir::quant::createConvertConstPass() {
  return new ConvertConstPass();
}

static PassRegistration<ConvertConstPass>
    pass("quant-convert-const",
         "Converts constants followed by qbarrier to actual quantized values");

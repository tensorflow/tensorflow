//===- RemoveInstrumentationPass.cpp - Removes instrumentation ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass to remove any instrumentation ops. It is often one
// of the final steps when performing quantization and is run after any
// decisions requiring instrumentation have been made.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Quantizer/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::quant;

namespace {

class RemoveInstrumentationPass
    : public FunctionPass<RemoveInstrumentationPass> {
  void runOnFunction() override;
};

template <typename OpTy>
class RemoveIdentityOpRewrite : public RewritePattern {
public:
  RemoveIdentityOpRewrite(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1);
    assert(op->getNumResults() == 1);

    rewriter.replaceOp(op, op->getOperand(0));
    return matchSuccess();
  }
};

} // end anonymous namespace

void RemoveInstrumentationPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  auto *context = &getContext();
  patterns.insert<RemoveIdentityOpRewrite<StatisticsOp>,
                  RemoveIdentityOpRewrite<StatisticsRefOp>,
                  RemoveIdentityOpRewrite<CoupledRefOp>>(context);
  applyPatternsGreedily(func, patterns);
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::quantizer::createRemoveInstrumentationPass() {
  return std::make_unique<RemoveInstrumentationPass>();
}

static PassRegistration<RemoveInstrumentationPass>
    pass("quantizer-remove-instrumentation",
         "Removes instrumentation and hints which have no effect on final "
         "execution");

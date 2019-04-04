//===- ConvertSimQuant.cpp - Converts simulated quant ops------------------===//
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Quantization/FakeQuantSupport.h"
#include "mlir/Quantization/Passes.h"
#include "mlir/Quantization/QuantOps.h"
#include "mlir/Quantization/UniformSupport.h"

using namespace mlir;
using namespace mlir::quant;

namespace {

class ConvertSimulatedQuantPass
    : public FunctionPass<ConvertSimulatedQuantPass> {
public:
  void runOnFunction() override;
};

} // end anonymous namespace

/// Rewrites ConstFakeQuant into a qbarrier/dbarrier pair.
class ConstFakeQuantRewrite : public RewritePattern {
public:
  bool *hadFailure;

  ConstFakeQuantRewrite(MLIRContext *context, bool *hadFailure)
      : RewritePattern(ConstFakeQuant::getOperationName(), 1, context),
        hadFailure(hadFailure) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // TODO: If this pattern comes up more frequently, consider adding core
    // support for failable rewrites.
    if (failableRewrite(op, rewriter)) {
      *hadFailure = true;
    }

    return matchSuccess();
  }

  bool failableRewrite(Operation *op, PatternRewriter &rewriter) const {
    auto fqOp = op->cast<ConstFakeQuant>();

    auto converter =
        ExpressedToUniformQuantizedConverter::forInputType(fqOp.getType());
    if (!converter) {
      return (op->emitError("unsupported quantized type conversion"), true);
    }

    UniformQuantizedType uniformElementType = fakeQuantAttrsToType(
        fqOp.getLoc(), fqOp.num_bits().getSExtValue(),
        fqOp.min().convertToFloat(), fqOp.max().convertToFloat(),
        fqOp.narrow_range(), converter.expressedType);

    if (!uniformElementType) {
      // Note that the fakeQuantAttrsToType will have emitted the error.
      return true;
    }

    Type quantizedType = converter.convert(uniformElementType);
    assert(quantizedType &&
           "Converter accepted a type that it did not convert");

    // TODO: Map to a qbarrier with an attribute like [Forced] to signal that
    // this is a forced/hard-coded constraint.
    auto qbarrier = rewriter.create<QuantizeBarrierOp>(
        op->getLoc(), quantizedType, fqOp.inputs());
    rewriter.replaceOpWithNewOp<DequantizeBarrierOp>(op, converter.inputType,
                                                     qbarrier.getResult());

    return false;
  }
};

void ConvertSimulatedQuantPass::runOnFunction() {
  bool hadFailure = false;
  OwningRewritePatternList patterns;
  auto &func = getFunction();
  auto *context = &getContext();
  patterns.push_back(
      llvm::make_unique<ConstFakeQuantRewrite>(context, &hadFailure));
  applyPatternsGreedily(func, std::move(patterns));
  if (hadFailure)
    signalPassFailure();
}

FunctionPassBase *createConvertSimulatedQuantPass() {
  return new ConvertSimulatedQuantPass();
}

static PassRegistration<ConvertSimulatedQuantPass>
    pass("quant-convert-simulated-quantization",
         "Converts training-time simulated quantization ops to corresponding "
         "quantize/dequantize casts.");

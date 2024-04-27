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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/Passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/UniformSupport.h"

using namespace mlir;
using namespace mlir::quantfork;

namespace {

#define GEN_PASS_DEF_QUANTCONVERTSIMULATEDQUANT
#include "tensorflow/compiler/mlir/lite/quantization/ir/Passes.h.inc"

struct ConvertSimulatedQuantPass
    : public impl::QuantConvertSimulatedQuantBase<ConvertSimulatedQuantPass> {
  void runOnOperation() override;
};

/// Base class rewrites ConstFakeQuant into a qbarrier/dbarrier pair.
template <typename ConcreteRewriteClass, typename FakeQuantOp>
class FakeQuantRewrite : public OpRewritePattern<FakeQuantOp> {
 public:
  using OpRewritePattern<FakeQuantOp>::OpRewritePattern;

  FakeQuantRewrite(MLIRContext *ctx, bool *hadFailure)
      : OpRewritePattern<FakeQuantOp>(ctx), hadFailure(hadFailure) {}

  LogicalResult matchAndRewrite(FakeQuantOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: If this pattern comes up more frequently, consider adding core
    // support for failable rewrites.
    if (failableRewrite(op, rewriter)) {
      *hadFailure = true;
      return failure();
    }

    return success();
  }

 private:
  bool *hadFailure;

  bool failableRewrite(FakeQuantOp op, PatternRewriter &rewriter) const {
    auto converter = ExpressedToQuantizedConverter::forInputType(op.getType());
    if (!converter) {
      return (op.emitError("unsupported quantized type conversion"), true);
    }

    quant::QuantizedType elementType =
        static_cast<const ConcreteRewriteClass *>(this)
            ->convertFakeQuantAttrsToType(op, converter.expressedType);

    if (!elementType) {
      // Note that the fakeQuantAttrsToType will have emitted the error.
      return true;
    }

    Type quantizedType = converter.convert(elementType);
    assert(quantizedType &&
           "Converter accepted a type that it did not convert");

    // TODO: Map to a qbarrier with an attribute like [Forced] to signal that
    // this is a forced/hard-coded constraint.
    auto qbarrier = rewriter.create<QuantizeCastOp>(op.getLoc(), quantizedType,
                                                    op.getInputs());
    rewriter.replaceOpWithNewOp<DequantizeCastOp>(op, converter.inputType,
                                                  qbarrier.getResult());

    return false;
  }
};

class ConstFakeQuantRewrite
    : public FakeQuantRewrite<ConstFakeQuantRewrite, ConstFakeQuant> {
 public:
  using BaseRewrite = FakeQuantRewrite<ConstFakeQuantRewrite, ConstFakeQuant>;

  ConstFakeQuantRewrite(MLIRContext *ctx, bool *hadFailure)
      : BaseRewrite(ctx, hadFailure) {}

  quant::QuantizedType convertFakeQuantAttrsToType(ConstFakeQuant fqOp,
                                                   Type expressedType) const {
    return fakeQuantAttrsToType(
        fqOp.getLoc(), fqOp.getNumBits(), fqOp.getMin().convertToFloat(),
        fqOp.getMax().convertToFloat(), fqOp.getNarrowRange(), expressedType,
        fqOp.getIsSigned());
  }
};

class ConstFakeQuantPerAxisRewrite
    : public FakeQuantRewrite<ConstFakeQuantPerAxisRewrite,
                              ConstFakeQuantPerAxis> {
 public:
  using BaseRewrite =
      FakeQuantRewrite<ConstFakeQuantPerAxisRewrite, ConstFakeQuantPerAxis>;

  ConstFakeQuantPerAxisRewrite(MLIRContext *ctx, bool *hadFailure)
      : BaseRewrite(ctx, hadFailure) {}

  quant::QuantizedType convertFakeQuantAttrsToType(ConstFakeQuantPerAxis fqOp,
                                                   Type expressedType) const {
    SmallVector<double, 4> min, max;
    min.reserve(fqOp.getMin().size());
    max.reserve(fqOp.getMax().size());
    for (auto m : fqOp.getMin())
      min.push_back(mlir::cast<FloatAttr>(m).getValueAsDouble());
    for (auto m : fqOp.getMax())
      max.push_back(mlir::cast<FloatAttr>(m).getValueAsDouble());

    return fakeQuantAttrsToType(fqOp.getLoc(), fqOp.getNumBits(),
                                fqOp.getAxis(), min, max, fqOp.getNarrowRange(),
                                expressedType, fqOp.getIsSigned());
  }
};

}  // namespace

void ConvertSimulatedQuantPass::runOnOperation() {
  bool hadFailure = false;
  auto func = getOperation();
  RewritePatternSet patterns(func.getContext());
  auto *ctx = func.getContext();
  patterns.add<ConstFakeQuantRewrite, ConstFakeQuantPerAxisRewrite>(
      ctx, &hadFailure);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  if (hadFailure) signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::quantfork::createConvertSimulatedQuantPass() {
  return std::make_unique<ConvertSimulatedQuantPass>();
}

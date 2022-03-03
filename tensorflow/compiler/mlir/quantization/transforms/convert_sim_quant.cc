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

// Converts simulated quant ops

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/ir/quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/transforms/pass_detail.h"
#include "tensorflow/compiler/mlir/quantization/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/utils/fake_quant_support.h"
#include "tensorflow/compiler/mlir/quantization/utils/uniform_support.h"

using namespace mlir;
using namespace mlir::quant;

namespace {
struct ConvertSimulatedQuantPass
    : public QuantConvertSimulatedQuantBase<ConvertSimulatedQuantPass> {
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

    QuantizedType elementType =
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
                                                    op.inputs());
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

  QuantizedType convertFakeQuantAttrsToType(ConstFakeQuant fqOp,
                                            Type expressedType) const {
    return fakeQuantAttrsToType(
        fqOp.getLoc(), fqOp.num_bits(), fqOp.min().convertToFloat(),
        fqOp.max().convertToFloat(), fqOp.narrow_range(), expressedType,
        fqOp.is_signed());
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

  QuantizedType convertFakeQuantAttrsToType(ConstFakeQuantPerAxis fqOp,
                                            Type expressedType) const {
    SmallVector<double, 4> min, max;
    min.reserve(fqOp.min().size());
    max.reserve(fqOp.max().size());
    for (auto m : fqOp.min())
      min.push_back(m.cast<FloatAttr>().getValueAsDouble());
    for (auto m : fqOp.max())
      max.push_back(m.cast<FloatAttr>().getValueAsDouble());

    return fakeQuantAttrsToType(fqOp.getLoc(), fqOp.num_bits(), fqOp.axis(),
                                min, max, fqOp.narrow_range(), expressedType,
                                fqOp.is_signed());
  }
};

} // namespace

void ConvertSimulatedQuantPass::runOnOperation() {
  bool hadFailure = false;
  auto func = getOperation();
  RewritePatternSet patterns(func.getContext());
  auto *ctx = func.getContext();
  patterns.add<ConstFakeQuantRewrite, ConstFakeQuantPerAxisRewrite>(
      ctx, &hadFailure);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  if (hadFailure)
    signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::quant::createConvertSimulatedQuantPass() {
  return std::make_unique<ConvertSimulatedQuantPass>();
}

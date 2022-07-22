// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Patterns to lower lmhlo.infeed/outfeed ops to xlir ops.

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct InfeedRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo::InfeedOp> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::InfeedOp>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::InfeedOp>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::InfeedOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    chain = rewriter.create<xla::gpu::InfeedOp>(
        op.getLoc(), chain.getType(), stream, adaptor.getOperands(), chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

struct OutfeedRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo::OutfeedOp> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::OutfeedOp>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::OutfeedOp>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::OutfeedOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    chain = rewriter.create<xla::gpu::OutfeedOp>(
        op.getLoc(), chain.getType(), stream, adaptor.getOperands(), chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateInfeedAndOutfeedConversionPattern(RewritePatternSet& patterns,
                                               TypeConverter& converter) {
  patterns.add<InfeedRewritePattern, OutfeedRewritePattern>(
      converter, patterns.getContext());
}

}  // namespace tensorflow

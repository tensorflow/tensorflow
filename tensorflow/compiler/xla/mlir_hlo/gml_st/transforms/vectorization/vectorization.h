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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_VECTORIZATION_VECTORIZATION_H
#define MLIR_HLO_GML_ST_TRANSFORMS_VECTORIZATION_VECTORIZATION_H

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace gml_st {

// The upper limit for vectorization of untiled `linalg.fill`. If a tensor has a
// static shape with more elements, then `linalg.fill` won't be vectorized. It
// is expected that such operations are tiled to get to small static shapes.
static constexpr int64_t kNumElementsThreshold = 1024;

// TODO(manany): This should be parameterized later on depending on hardware.
static constexpr int64_t kNumElementsVectorization = 8;

template <typename OpTy>
struct VectorizationPattern : public mlir::OpRewritePattern<OpTy> {
  VectorizationPattern(MLIRContext *context,
                       llvm::function_ref<bool(OpTy)> matchFn,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<OpTy>(context, benefit), filterFn(matchFn) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");
    return mlir::linalg::vectorize(rewriter, op);
  }

 private:
  llvm::function_ref<bool(OpTy)> filterFn;
};

void populateTransferReadOfOneDimExpandShapePattern(
    RewritePatternSet &patterns);

RewritePatternSet getDefaultVectorizationPatterns(MLIRContext *ctx);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_VECTORIZATION_VECTORIZATION_H

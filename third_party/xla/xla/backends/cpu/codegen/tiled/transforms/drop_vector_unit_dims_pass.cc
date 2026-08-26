/* Copyright 2026 The OpenXLA Authors.

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

#include <utility>

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DEF_DROPVECTORUNITDIMSPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

class DropVectorUnitDimsPass
    : public impl::DropVectorUnitDimsPassBase<DropVectorUnitDimsPass> {
 public:
  using DropVectorUnitDimsPassBase::DropVectorUnitDimsPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    mlir::vector::populateDropUnitDimWithShapeCastPatterns(patterns);
    mlir::vector::populateDropInnerMostUnitDimsXferOpPatterns(patterns);
    mlir::vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu

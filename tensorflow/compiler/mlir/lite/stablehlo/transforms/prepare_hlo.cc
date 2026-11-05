/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/transforms/optimization/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/fft.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_PREPAREHLOPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

class PrepareHloPass : public impl::PrepareHloPassBase<PrepareHloPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareHloPass);

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_prepare_hlo.inc"

// Removes `stablehlo.pad` ops that pad nothing, e.g. the remainder left by
// `PullOutNegativePads` when all pads were negative. Unlike `mhlo.pad`,
// `stablehlo.pad` has no folder that would drop these.
class RemoveNoopPad : public OpRewritePattern<stablehlo::PadOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::PadOp op,
                                PatternRewriter& rewriter) const override {
    auto is_zero = [](int64_t v) { return v == 0; };
    if (!llvm::all_of(op.getEdgePaddingLow(), is_zero) ||
        !llvm::all_of(op.getEdgePaddingHigh(), is_zero) ||
        !llvm::all_of(op.getInteriorPadding(), is_zero) ||
        op.getOperand().getType() != op.getType()) {
      return failure();
    }
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

// Folds `stablehlo.transpose` of a splat constant, like `mhlo.transpose`'s
// folder did. Non-splat constants are deliberately not folded; see below.
class FoldSplatTranspose : public OpRewritePattern<stablehlo::TransposeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    SplatElementsAttr splat;
    if (!op.getType().hasStaticShape() ||
        !matchPattern(op.getOperand(), m_Constant(&splat))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, splat.reshape(op.getType()));
    return success();
  }
};

void PrepareHloPass::runOnOperation() {
  MLIRContext* context = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.add<RemoveNoopPad, FoldSplatTranspose>(context);

  // MHLO ops carried folders and canonicalization patterns that ran implicitly
  // in every rewrite driver (folding converts/broadcasts of constants,
  // cancelling transpose pairs, ...). StableHLO ops mostly don't, so add the
  // equivalent StableHLO folding and simplification patterns here, right
  // before legalization; the legalization patterns rely on that canonical form
  // (e.g. reduce_sum / constant -> tfl.mean). Folder float rewrites that may
  // change numerics are disabled.
  stablehlo::StablehloAggressiveFolderPassOptions folder_options;
  folder_options.optimizeFloat = false;
  stablehlo::populateStablehloAggressiveFolderPatterns(context, &patterns,
                                                       folder_options);
  // Unlike `mhlo.transpose` (which only folded splats and identity
  // permutations), StableHLO's transpose folder materializes the transpose of
  // any constant element by element, ignoring `foldOpElementLimit`. On large
  // weights (e.g. `F.linear` kernels) that dominates conversion time, so drop
  // it and keep MHLO's behavior: splats are folded by `FoldSplatTranspose` and
  // identity permutations by the canonicalization patterns below.
  llvm::erase_if(
      patterns.getNativePatterns(),
      [](const std::unique_ptr<RewritePattern>& pattern) {
        return pattern->getDebugName().ends_with("::FoldTransposeOpPattern");
      });
  stablehlo::populateStablehloCanonicalizationPatterns(context, &patterns);

  PopulatePrepareConvPatterns(context, patterns);
  PopulatePrepareReduceWindowPatterns(context, patterns);
  PopulatePrepareSlicePatterns(context, patterns);
  PopulateCustomCallPreparePatterns(context, patterns);
  PopulatePrepareFftPatterns(context, patterns);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareHloPass() {
  return std::make_unique<PrepareHloPass>();
}

static PassRegistration<PrepareHloPass> pass;

}  // namespace odml
}  // namespace mlir

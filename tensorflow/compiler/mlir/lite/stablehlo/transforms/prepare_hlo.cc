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
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/fft.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"

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
        !matchPattern(op.getOperand(), m_Constant(&splat)) ||
        splat.getElementType() != op.getType().getElementType()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, splat.reshape(op.getType()));
    return success();
  }
};

// Folds `stablehlo.convert` of a constant to a float type, like
// `mhlo.convert`'s folder did (with the same 65536-element limit). StableHLO's
// folder only folds float converts with `optimizeFloat`, which also enables
// numerics-changing rewrites. E.g. folding a f16 bias constant converted to f32
// lets the TFL bias fusion into FULLY_CONNECTED kick in.
class FoldConstantToFloatConvert
    : public OpRewritePattern<stablehlo::ConvertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter& rewriter) const override {
    constexpr int64_t kFoldOpElementLimit = 65536;
    auto result_type = op.getType();
    auto result_element_type =
        dyn_cast<FloatType>(result_type.getElementType());
    DenseElementsAttr attr;
    if (!result_element_type || !result_type.hasStaticShape() ||
        !matchPattern(op.getOperand(), m_Constant(&attr)) ||
        // Not a quantized constant (that would be a dequantize).
        attr.getElementType() !=
            getElementTypeOrSelf(op.getOperand().getType()) ||
        attr.getNumElements() > kFoldOpElementLimit) {
      return failure();
    }
    const llvm::fltSemantics& semantics =
        result_element_type.getFloatSemantics();
    DenseElementsAttr folded;
    if (auto float_attr = dyn_cast<DenseFPElementsAttr>(attr)) {
      folded =
          float_attr.mapValues(result_element_type, [&](const APFloat& value) {
            APFloat converted = value;
            bool loses_info;
            converted.convert(semantics, APFloat::rmNearestTiesToEven,
                              &loses_info);
            return converted.bitcastToAPInt();
          });
    } else if (auto int_attr = dyn_cast<DenseIntElementsAttr>(attr)) {
      auto int_type = cast<IntegerType>(int_attr.getElementType());
      const bool is_signed = !int_type.isUnsigned() && int_type.getWidth() != 1;
      folded = int_attr.mapValues(result_element_type, [&](const APInt& value) {
        APFloat converted(semantics);
        converted.convertFromAPInt(value, is_signed,
                                   APFloat::rmNearestTiesToEven);
        return converted.bitcastToAPInt();
      });
    } else {
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, folded);
    return success();
  }
};

// Folds `stablehlo.reshape` of a constant, like `mhlo.reshape`'s folder did.
// StableHLO's folder only folds float reshapes with `optimizeFloat`.
class FoldConstantReshape : public OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    DenseElementsAttr attr;
    if (!op.getType().hasStaticShape() ||
        !matchPattern(op.getOperand(), m_Constant(&attr)) ||
        attr.getElementType() != op.getType().getElementType()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, attr.reshape(op.getType()));
    return success();
  }
};

void PrepareHloPass::runOnOperation() {
  MLIRContext* context = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.add<RemoveNoopPad>(context);
  // MHLO's folders ran implicitly in every rewrite driver, including on the
  // ops created by the patterns below. StableHLO's ops mostly have no folders,
  // so these patterns provide the equivalent folding of constants. The broader
  // StableHLO folding and simplification runs separately in SimplifyHloPass.
  patterns
      .add<FoldSplatTranspose, FoldConstantToFloatConvert, FoldConstantReshape>(
          context);
  patterns.add(FoldSplatBroadcastInDim);

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

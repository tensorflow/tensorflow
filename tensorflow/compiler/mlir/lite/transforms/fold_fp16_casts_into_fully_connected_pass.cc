/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_FOLDFP16CASTSINTOFULLYCONNECTEDPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Returns `value`'s type if it is a ranked tensor of exactly `element_type`.
RankedTensorType GetRankedTensorOf(Value value, Type element_type) {
  auto type = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getElementType() != element_type) return nullptr;
  return type;
}

// Folds the f16 <-> f32 casts around a float fully_connected:
//
//   %x32 = tfl.cast(%x16) : f16 -> f32
//   %y32 = tfl.fully_connected(%x32, %filter, %bias) : f32
//   %y16 = tfl.cast(%y32) : f32 -> f16
//
// into
//
//   %y16 = tfl.fully_connected(%x16, %filter, %bias) : f16
//
// The filter and bias keep their types, so the result is a mixed precision
// fully_connected (f16 activations, f32 or quantized weights). Only runtimes
// that support that combination can execute the result, which is why the pass
// is opt-in.
struct FoldFp16CastsIntoFullyConnectedPattern
    : public OpRewritePattern<TFL::CastOp> {
  using OpRewritePattern<TFL::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::CastOp output_cast,
                                PatternRewriter& rewriter) const override {
    const Type f16 = rewriter.getF16Type();
    const Type f32 = rewriter.getF32Type();

    if (!GetRankedTensorOf(output_cast.getOutput(), f16)) return failure();
    Value fc_output = output_cast.getInput();
    if (!GetRankedTensorOf(fc_output, f32)) return failure();

    auto fc = fc_output.getDefiningOp<TFL::FullyConnectedOp>();
    // A shuffled-weights fully_connected has a second result and cannot be
    // replaced by the single-result op built below.
    if (!fc || fc.getNumResults() != 1) return failure();
    if (fc.getWeightsFormat() != "DEFAULT") return failure();
    // Other users still need the f32 result.
    if (!fc_output.hasOneUse()) return failure();

    auto input_cast = fc.getInput().getDefiningOp<TFL::CastOp>();
    if (!input_cast) return failure();
    Value input = input_cast.getInput();
    if (!GetRankedTensorOf(input, f16)) return failure();
    if (!GetRankedTensorOf(input_cast.getOutput(), f32)) return failure();

    auto fc_output_type = mlir::cast<RankedTensorType>(fc_output.getType());
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        fc.getLoc(), fc_output_type.clone(f16), input, fc.getFilter(),
        fc.getBias(), fc.getFusedActivationFunctionAttr(),
        fc.getWeightsFormatAttr(), fc.getKeepNumDimsAttr(),
        fc.getAsymmetricQuantizeInputsAttr());

    rewriter.replaceOp(output_cast, new_fc.getOutput());
    rewriter.eraseOp(fc);
    // `input_cast` may still feed other ops; it is dead code otherwise and is
    // erased by the greedy driver.
    return success();
  }
};

struct FoldFp16CastsIntoFullyConnectedPass
    : public impl::FoldFp16CastsIntoFullyConnectedPassBase<
          FoldFp16CastsIntoFullyConnectedPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FoldFp16CastsIntoFullyConnectedPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* context = func.getContext();

    RewritePatternSet patterns(context);
    patterns.add<FoldFp16CastsIntoFullyConnectedPattern>(context);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateFoldFp16CastsIntoFullyConnectedPass() {
  return std::make_unique<FoldFp16CastsIntoFullyConnectedPass>();
}

}  // namespace TFL
}  // namespace mlir

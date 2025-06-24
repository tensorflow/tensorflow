/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for legalizing HLO to TensorFlow.

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Utils/IndexingUtils.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir {
namespace odml {
namespace {

#define DEBUG_TYPE "stablehlo-optimize-layout"

#define GEN_PASS_DEF_TRANSPOSECOMMUTEOPSPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

class TransposeCommuteOpsPass
    : public impl::TransposeCommuteOpsPassBase<TransposeCommuteOpsPass> {
  void runOnOperation() override;
};

// Inversely permutate a given vector
static SmallVector<int64_t> InvertPermutationToVector(ArrayRef<int64_t> vec,
                                                      ArrayRef<int64_t> perm) {
  return applyPermutation(vec, invertPermutationVector(perm));
}

static RankedTensorType GetPermutedTensorTypeHelper(RankedTensorType type,
                                                    ArrayRef<int64_t> perm,
                                                    bool isInvert) {
  SmallVector<int64_t, 4> permutedShape = applyPermutation(
      type.getShape(), isInvert ? invertPermutationVector(perm) : perm);
  return RankedTensorType::get(permutedShape, type.getElementType());
}

static RankedTensorType GetInvertPermutedTensorType(RankedTensorType type,
                                                    ArrayRef<int64_t> perm) {
  return GetPermutedTensorTypeHelper(type, perm, true /*isInvert*/);
}

static Value CreateTranspose(OpBuilder& builder, Value source,
                             ArrayRef<int64_t> perm) {
  return builder.create<stablehlo::TransposeOp>(source.getLoc(), source, perm)
      ->getResult(0);
}

// Transform pad(transpose(x)) to transpose(pad(x))
struct TransposeCommuteWithPad : public OpRewritePattern<stablehlo::PadOp> {
  using OpRewritePattern<stablehlo::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::PadOp pad_op,
                                PatternRewriter& rewriter) const override {
    Value pad_input = pad_op.getOperand();
    RankedTensorType pad_type = mlir::cast<RankedTensorType>(pad_op.getType());

    auto transpose_op = pad_input.getDefiningOp<stablehlo::TransposeOp>();
    if (!transpose_op || !transpose_op->hasOneUse()) return failure();
    Value transpose_input = transpose_op.getOperand();

    ArrayRef<int64_t> transpose_perm = transpose_op.getPermutation();
    SmallVector<int64_t> new_padding_low =
        InvertPermutationToVector(pad_op.getEdgePaddingLow(), transpose_perm);
    SmallVector<int64_t> new_padding_high =
        InvertPermutationToVector(pad_op.getEdgePaddingHigh(), transpose_perm);
    SmallVector<int64_t> new_padding_interrier =
        InvertPermutationToVector(pad_op.getInteriorPadding(), transpose_perm);

    RankedTensorType new_pad_type =
        GetInvertPermutedTensorType(pad_type, transpose_perm);
    Value new_pad = rewriter.create<stablehlo::PadOp>(
        pad_op.getLoc(), new_pad_type, transpose_input,
        pad_op.getPaddingValue(), new_padding_low, new_padding_high,
        new_padding_interrier);

    Value orig_pad = CreateTranspose(rewriter, new_pad, transpose_perm);
    rewriter.replaceOp(pad_op, orig_pad);
    return success();
  }
};

// Transform reduce_window(transpose(x)) to transpose(reduce_window(x))
struct TransposeCommuteWithReduceWindow
    : public OpRewritePattern<stablehlo::ReduceWindowOp> {
  using OpRewritePattern<stablehlo::ReduceWindowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReduceWindowOp reduce_op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = reduce_op.getContext();
    ValueRange inputs = reduce_op.getInputs();
    // Only handle binary reduce ops for now
    if (inputs.size() != 1) return failure();
    Value reduce_input = inputs[0];

    RankedTensorType reduce_type =
        mlir::cast<RankedTensorType>(reduce_op.getResultTypes()[0]);

    auto transpose_op = reduce_input.getDefiningOp<stablehlo::TransposeOp>();
    if (!transpose_op || !transpose_op->hasOneUse()) return failure();
    Value transpose_input = transpose_op.getOperand();

    ArrayRef<int64_t> transpose_perm = transpose_op.getPermutation();

    // Inversely transposes all the attributes to prepare for the new reduce op
    auto getInvertPermutedAttr =
        [&](std::optional<ArrayRef<int64_t>> vals) -> DenseI64ArrayAttr {
      return vals.has_value()
                 ? DenseI64ArrayAttr::get(
                       ctx, InvertPermutationToVector(*vals, transpose_perm))
                 : nullptr;
    };
    DenseI64ArrayAttr new_window_dimensions =
        getInvertPermutedAttr(reduce_op.getWindowDimensions());
    DenseI64ArrayAttr new_window_strides =
        getInvertPermutedAttr(reduce_op.getWindowStrides());
    DenseI64ArrayAttr new_base_dilations =
        getInvertPermutedAttr(reduce_op.getBaseDilations());
    DenseI64ArrayAttr new_win_dilations =
        getInvertPermutedAttr(reduce_op.getWindowDilations());

    auto padding = reduce_op.getPadding();
    int64_t rank = transpose_perm.size();
    DenseIntElementsAttr new_padding_attr = nullptr;
    if (padding.has_value()) {
      SmallVector<int64_t> new_padding(rank * 2, 0);
      auto old_padding = (*padding).getValues<int64_t>();
      for (int64_t idx = 0; idx < rank; ++idx) {
        new_padding[2 * transpose_perm[idx]] = old_padding[2 * idx];
        new_padding[2 * transpose_perm[idx] + 1] = old_padding[2 * idx + 1];
      }
      new_padding_attr =
          DenseIntElementsAttr::get((*padding).getType(), new_padding);
    }

    RankedTensorType new_reduce_type =
        GetInvertPermutedTensorType(reduce_type, transpose_perm);
    auto new_reduce_op = rewriter.create<stablehlo::ReduceWindowOp>(
        reduce_op.getLoc(), new_reduce_type, transpose_input,
        reduce_op.getInitValues()[0], new_window_dimensions, new_window_strides,
        new_base_dilations, new_win_dilations, new_padding_attr);
    IRMapping mapping;
    reduce_op.getBody().cloneInto(&new_reduce_op.getBody(), mapping);

    Value orig_reduce_op =
        CreateTranspose(rewriter, new_reduce_op->getResult(0), transpose_perm);
    rewriter.replaceOp(reduce_op, orig_reduce_op);
    return success();
  }
};

void TransposeCommuteOpsPass::runOnOperation() {
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<TransposeCommuteWithPad, TransposeCommuteWithReduceWindow>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTransposeCommuteOpsPass() {
  return std::make_unique<TransposeCommuteOpsPass>();
}

static PassRegistration<TransposeCommuteOpsPass> pass;

}  // end namespace odml
}  // end namespace mlir

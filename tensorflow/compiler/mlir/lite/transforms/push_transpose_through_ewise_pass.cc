/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/push_transpose_through_ewise_pass.h"

#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {

// Compute the permutation that would take `arr` to the identity.
llvm::SmallVector<int32_t> InvertPermutation(llvm::SmallVector<int32_t> arr) {
  llvm::SmallVector<int32_t> inverse_arr(arr.size());
  for (int32_t i = 0; i < arr.size(); ++i) {
    inverse_arr[arr[i]] = i;
  }
  return inverse_arr;
}

llvm::SmallVector<int64_t> PermuteShape(llvm::ArrayRef<int64_t> shape,
                                        llvm::ArrayRef<int32_t> perm) {
  llvm::SmallVector<int64_t> new_shape(shape.size());
  for (const auto &perm_element : enumerate(perm)) {
    new_shape[perm_element.index()] = shape[perm_element.value()];
  }
  return new_shape;
}

// Determine if op commutes with transposes. Requires a strict
// definition of Elementwise, all i/o shapes and types must be same-rank
// broadcastable. Consider moving this into attribute later.
bool IsElementwise(Operation *op) {
  if (!(llvm::isa<TFL::AddOp, TFL::MulOp, TFL::DivOp, TFL::SubOp,
                  TFL::MaximumOp, TFL::MinimumOp>(op))) {
    return false;
  }

  auto opr1_type =
      llvm::dyn_cast_or_null<RankedTensorType>(op->getOperand(0).getType());
  auto opr2_type =
      llvm::dyn_cast_or_null<RankedTensorType>(op->getOperand(1).getType());
  auto res_type =
      llvm::dyn_cast_or_null<RankedTensorType>(op->getResult(0).getType());

  if (!(opr1_type && opr2_type && res_type)) {
    return false;
  }

  return true;
}

// In some cases, transposes may commute with elementwise operations. In order
// to make as many tranposes redudant as possible, we can "push" transposes
// back so that they fuse later on. These patterns handles 2 such cases in
// a conservative fashion; on-net it will never add to the number of transposes
// in the graph.

// ewise(tpose(x), tpose(y)) -> tpose(ewise(x, y))
//    iff tpose(x) or tpose(y) has one use and have same permutation.
//
// Proof:
// WLOG, let tpose(x) have 1 use. Then ewise is the only user, and removing
// its use of tpose(x) will render tpose(x) deadcode. So in this case
// we both remove 1 and add 1 transpose to the graph thus the number remains
// unchanged.
class CommuteBothInputsTransposedWithEwiseOps : public RewritePattern {
 public:
  explicit CommuteBothInputsTransposedWithEwiseOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!IsElementwise(op)) {
      return failure();
    }

    const bool any_blargs = llvm::any_of(
        op->getOperands(),
        [](Value opr) -> bool { return llvm::isa<BlockArgument>(opr); });
    if (any_blargs) {
      return failure();
    }

    auto tpose_arg1 = llvm::dyn_cast_or_null<TFL::TransposeOp>(
        op->getOperand(0).getDefiningOp());
    auto tpose_arg2 = llvm::dyn_cast_or_null<TFL::TransposeOp>(
        op->getOperand(1).getDefiningOp());

    if (!tpose_arg1 || !tpose_arg2) {
      return failure();
    }

    auto tpose_arg1_type =
        llvm::dyn_cast<RankedTensorType>(tpose_arg1->getResultTypes()[0]);
    auto tpose_arg2_type =
        llvm::dyn_cast<RankedTensorType>(tpose_arg2->getResultTypes()[0]);
    if (tpose_arg1_type.getRank() != tpose_arg2_type.getRank()) {
      return failure();
    }

    if (llvm::isa<BlockArgument>(tpose_arg1.getPerm()) ||
        llvm::isa<BlockArgument>(tpose_arg2.getPerm())) {
      return failure();
    }
    auto perm1 = llvm::dyn_cast_or_null<arith::ConstantOp>(
        tpose_arg1.getPerm().getDefiningOp());
    auto perm2 = llvm::dyn_cast_or_null<arith::ConstantOp>(
        tpose_arg2.getPerm().getDefiningOp());
    if (!perm1 || !perm2) {
      return failure();
    }

    auto perm1_value = llvm::dyn_cast<DenseElementsAttr>(perm1.getValue())
                           .getValues<int32_t>();
    auto perm2_value = llvm::dyn_cast<DenseElementsAttr>(perm2.getValue())
                           .getValues<int32_t>();

    llvm::SmallVector<int32_t> perm1_arr(perm1_value.begin(),
                                         perm1_value.end());
    llvm::SmallVector<int32_t> perm2_arr(perm2_value.begin(),
                                         perm2_value.end());
    if (perm1_arr != perm2_arr) {
      return failure();
    }

    // Compute inverse of input transpose.
    llvm::SmallVector<int32_t> inverse_perm_arr = InvertPermutation(perm1_arr);

    if (!(tpose_arg1->hasOneUse() || tpose_arg2->hasOneUse())) {
      return failure();
    }

    auto current_out_type =
        llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto new_out_type = RankedTensorType::get(
        PermuteShape(current_out_type.getShape(), inverse_perm_arr),
        current_out_type.getElementType());

    // Create new ewise op to appear before the tranpose.
    auto *new_ewise_op =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        {tpose_arg1.getOperand(0), tpose_arg2.getOperand(0)},
                        new_out_type, op->getAttrs());

    // Apply original tranpose to output of ewise op.
    auto out_tpose_op = rewriter.create<TFL::TransposeOp>(
        new_ewise_op->getLoc(), op->getResult(0).getType(),
        new_ewise_op->getResults()[0], perm1);
    rewriter.replaceOp(op, out_tpose_op.getOperation());
    return success();
  }
};

// ewise(tpose(x), y) -> tpose(ewise(x, tpose^{-1}(y)))
//  iff y is const and tpose(x) has one use.
//
// Proof:
// Since y is const, tpose(y) will be folded at compile time, since
// tpose(x) has one use it will be DCEd, #tposes unchanged.
class CommuteTransposeWithEwiseOps : public RewritePattern {
 public:
  explicit CommuteTransposeWithEwiseOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!IsElementwise(op)) {
      return failure();
    }

    const bool any_blargs = llvm::any_of(
        op->getOperands(),
        [](Value opr) -> bool { return llvm::isa<BlockArgument>(opr); });
    if (any_blargs) {
      return failure();
    }

    TFL::TransposeOp tpose_arg = nullptr;
    Operation *cst_arg = nullptr;

    for (Value opr : op->getOperands()) {
      auto *defining_op = opr.getDefiningOp();
      if (llvm::isa<TFL::TransposeOp>(defining_op) && !tpose_arg) {
        tpose_arg = llvm::dyn_cast<TFL::TransposeOp>(defining_op);
        continue;
      }
      if (defining_op->hasTrait<OpTrait::ConstantLike>()) {
        cst_arg = defining_op;
      }
    }
    if (!tpose_arg || !cst_arg) {
      return failure();
    }

    if (llvm::isa<BlockArgument>(tpose_arg.getPerm())) {
      return failure();
    }

    if (!tpose_arg->hasOneUse()) {
      return failure();
    }

    auto tpose_arg_type =
        llvm::dyn_cast<RankedTensorType>(tpose_arg->getResultTypes()[0]);
    auto cst_arg_type =
        llvm::dyn_cast<RankedTensorType>(cst_arg->getResultTypes()[0]);

    auto tpose_arg_rank = tpose_arg_type.getRank();
    auto cst_arg_rank = cst_arg_type.getRank();

    if (!(tpose_arg_rank == cst_arg_rank || cst_arg_rank == 0 ||
          tpose_arg_rank == 0)) {
      return failure();
    }

    auto perm = llvm::dyn_cast_or_null<arith::ConstantOp>(
        tpose_arg.getPerm().getDefiningOp());
    if (!perm) {
      return failure();
    }
    auto perm_value =
        llvm::dyn_cast<DenseElementsAttr>(perm.getValue()).getValues<int32_t>();
    llvm::SmallVector<int32_t> perm_arr(perm_value.begin(), perm_value.end());

    // Compute inverse of input transpose.
    llvm::SmallVector<int32_t> inverse_perm = InvertPermutation(perm_arr);

    if (tpose_arg.getResult().getType() != op->getResult(0).getType()) {
      // If types are not equal here than the transpose is taking place
      // before an implicit broadcast in the elementwise. We need
      // to be conservative and not move the tranpose here, since
      // we may be transposing a larger tensor.
      return failure();
    }

    auto other_input_type =
        mlir::cast<RankedTensorType>(cst_arg->getResult(0).getType());

    Operation *tposed_const;
    if (other_input_type.getNumElements() == 1) {
      tposed_const = cst_arg;
    } else {
      auto inverse_perm_attr = DenseIntElementsAttr::get(
          RankedTensorType::get(inverse_perm.size(), rewriter.getI32Type()),
          inverse_perm);
      auto inverse_perm_op =
          rewriter.create<arith::ConstantOp>(perm.getLoc(), inverse_perm_attr);

      // Transpose the input constant.
      auto in_rtt =
          llvm::dyn_cast<RankedTensorType>(cst_arg->getResult(0).getType());

      auto inverse_type =
          RankedTensorType::get(PermuteShape(in_rtt.getShape(), inverse_perm),
                                in_rtt.getElementType());

      tposed_const = rewriter.create<TFL::TransposeOp>(
          cst_arg->getLoc(), inverse_type, cst_arg->getResult(0),
          inverse_perm_op);
    }

    auto current_out_type =
        llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto new_out_type = RankedTensorType::get(
        PermuteShape(current_out_type.getShape(), inverse_perm),
        current_out_type.getElementType());

    // Create new ewise op to appear before the tranpose.
    auto *new_ewise_op =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        {tpose_arg.getInput(), tposed_const->getResult(0)},
                        new_out_type, op->getAttrs());

    // Apply original tranpose to output of ewise op.
    auto out_tpose_op = rewriter.create<TFL::TransposeOp>(
        new_ewise_op->getLoc(), op->getResult(0).getType(),
        new_ewise_op->getResults()[0], perm);
    rewriter.replaceOp(op, out_tpose_op.getOperation());
    return success();
  }
};
}  // namespace

void PushTransposeThroughEwisePass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());

  patterns.add<CommuteTransposeWithEwiseOps>(&getContext());
  patterns.add<CommuteBothInputsTransposedWithEwiseOps>(&getContext());
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace TFL
}  // namespace mlir

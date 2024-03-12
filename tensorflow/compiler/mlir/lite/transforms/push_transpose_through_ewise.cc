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

#include <cstdint>
#include <memory>
#include <optional>
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
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_PUSHTRANSPOSETHROUGHEWISEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

class PushTransposeThroughEwisePass
    : public impl::PushTransposeThroughEwisePassBase<
          PushTransposeThroughEwisePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PushTransposeThroughEwisePass)
  void runOnOperation() override;
};

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
// broadcastable and fully static. Consider moving this into attribute later.
bool IsElementwise(Operation *op) {
  if (!(llvm::isa<TFL::AddOp, TFL::MulOp, TFL::DivOp, TFL::SubOp>(op))) {
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

  if (!(opr1_type.getRank() == opr2_type.getRank() &&
        opr1_type.getRank() == res_type.getRank())) {
    return false;
  }

  if (!opr1_type.hasStaticShape() && opr2_type.hasStaticShape() &&
      res_type.hasStaticShape()) {
    return false;
  }

  return true;
}

// In some cases, transposes may commute with elementwise operations. In order
// to make as many tranposes redudant as possible, we can "push" transposes
// back so that they fuse later on. This pattern handles 2 such cases in
// a conservative fashion; on-net it will never add to the number of transposes
// in the graph.
//
// Case 1:
// -------
//
// ewise(tpose(x), y) -> tpose(ewise(x, tpose^{-1}(y)))
//  iff y is const
//
// Proof:
// Since y is const, tpose(y) will be folded at compile time, #tposes unchanged.
//
// Case 2:
// -------
//
// ewise(tpose(x), tpose(y)) -> tpose(ewise(x, y))
//    iff tpose(x) or tpose(y) has one use and have same permutation.
//
// Proof:
// WLOG, let tpose(x) have 1 use. Then ewise is the only user, and removing
// its use of tpose(x) will render tpose(x) deadcode. So in this case
// we both remove 1 and add 1 transpose to the graph thus the number remains
// unchanged.
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

    std::optional<Value> tpose_arg;
    std::optional<Value> other_arg;

    for (Value opr : op->getOperands()) {
      auto *defining_op = opr.getDefiningOp();
      if (llvm::isa<TFL::TransposeOp>(defining_op) && !tpose_arg.has_value()) {
        tpose_arg = opr;
        continue;
      }
      other_arg = opr;
    }

    if (!tpose_arg.has_value() || !other_arg.has_value()) {
      return failure();
    }

    Operation *opq_tpose_op = tpose_arg.value().getDefiningOp();
    Operation *opq_other_op = other_arg.value().getDefiningOp();
    Value other_op_result;

    auto tpose_op = llvm::dyn_cast<TFL::TransposeOp>(opq_tpose_op);

    auto perm = llvm::dyn_cast_or_null<arith::ConstantOp>(
        tpose_op.getPerm().getDefiningOp());
    if (!perm) {
      return failure();
    }
    auto perm_value =
        llvm::dyn_cast<DenseElementsAttr>(perm.getValue()).getValues<int32_t>();
    llvm::SmallVector<int32_t> perm_arr(perm_value.begin(), perm_value.end());

    // Compute inverse of input transpose.
    llvm::SmallVector<int32_t> inverse_perm = InvertPermutation(perm_arr);

    if (opq_other_op->hasTrait<OpTrait::ConstantLike>()) {
      // Case 1

      if (tpose_op.getResult().getType() != op->getResult(0).getType()) {
        // If types are not equal here than the transpose is taking place
        // before an implicit broadcast in the elementwise. We need
        // to be conservative and not move the tranpose here, since
        // we may be transposing a larger tensor.
        return failure();
      }

      auto inverse_perm_attr = DenseIntElementsAttr::get(
          RankedTensorType::get(inverse_perm.size(), rewriter.getI32Type()),
          inverse_perm);
      auto inverse_perm_op =
          rewriter.create<arith::ConstantOp>(perm.getLoc(), inverse_perm_attr);

      // Transpose the input constant.
      auto in_rtt = llvm::dyn_cast_or_null<RankedTensorType>(
          opq_other_op->getResult(0).getType());
      if (!in_rtt) {
        return failure();
      }
      auto inverse_type =
          RankedTensorType::get(PermuteShape(in_rtt.getShape(), inverse_perm),
                                in_rtt.getElementType());
      auto tposed_const = rewriter.create<TFL::TransposeOp>(
          opq_other_op->getLoc(), inverse_type, opq_other_op->getResult(0),
          inverse_perm_op);

      other_op_result = tposed_const.getResult();

    } else if (auto other_tpose_op =
                   llvm::dyn_cast_or_null<TFL::TransposeOp>(opq_other_op)) {
      // Case 2

      auto other_perm = llvm::dyn_cast_or_null<arith::ConstantOp>(
          other_tpose_op.getPerm().getDefiningOp());
      if (!other_perm) {
        return failure();
      }
      if (other_perm.getValue() != perm.getValue()) {
        return failure();
      }
      if (!(other_tpose_op->hasOneUse() || tpose_op->hasOneUse())) {
        return failure();
      }

      other_op_result = other_tpose_op.getInput();
    } else {
      return failure();
    }

    auto current_out_type =
        llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto new_out_type = RankedTensorType::get(
        PermuteShape(current_out_type.getShape(), inverse_perm),
        current_out_type.getElementType());

    // Create new ewise op to appear before the tranpose.
    auto *new_ewise_op =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        {tpose_op.getOperand(0), other_op_result}, new_out_type,
                        op->getAttrs());

    // Apply original tranpose to output of ewise op.
    auto out_tpose_op = rewriter.create<TFL::TransposeOp>(
        new_ewise_op->getLoc(), op->getResult(0).getType(),
        new_ewise_op->getResults()[0], perm);
    rewriter.replaceOp(op, out_tpose_op.getOperation());
    return success();
  }
};

void PushTransposeThroughEwisePass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());

  patterns.add<CommuteTransposeWithEwiseOps>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePushTransposeThroughEwisePass() {
  return std::make_unique<PushTransposeThroughEwisePass>();
}

}  // namespace TFL
}  // namespace mlir

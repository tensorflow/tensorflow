/* Copyright 2025 The OpenXLA Authors.

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
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAFOLDTRANSPOSEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

template <typename T>
auto ApplyPermutation(T input, ArrayRef<int32_t> perm) {
  SmallVector<std::decay_t<decltype(*input.begin())>> result;
  result.reserve(perm.size());
  for (int32_t p : perm) {
    result.push_back(input[p]);
  }
  return result;
}

LogicalResult FoldTransposeOfLoad(TransOp op, PatternRewriter& rewriter) {
  auto load = op.getSrc().getDefiningOp<LoadOp>();
  if (!load) {
    return rewriter.notifyMatchFailure(op, "Transpose source is not a load.");
  }
  auto make_ptr = load.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_ptr) {
    return rewriter.notifyMatchFailure(op, "Expected load of make_tensor_ptr.");
  }
  if (load.getMask() || load.getOther()) {
    return rewriter.notifyMatchFailure(op, "Unsupported load.");
  }

  auto apply_order = [&](auto range) {
    return ApplyPermutation(range, op.getOrder());
  };

  auto ptr_type =
      PointerType::get(op.getType(), make_ptr.getType().getAddressSpace());
  auto new_make_ptr = rewriter.create<MakeTensorPtrOp>(
      make_ptr.getLoc(), ptr_type, make_ptr.getBase(),
      apply_order(make_ptr.getShape()), apply_order(make_ptr.getStrides()),
      // Leave original order, it's unused but checked to be default elsewhere.
      apply_order(make_ptr.getOffsets()), make_ptr.getOrderAttr());

  SmallVector<bool> boundary_check_bits(op.getType().getRank());
  for (auto dim : load.getBoundaryCheck()) {
    boundary_check_bits[dim] = true;
  }
  SmallVector<int32_t> new_boundary_check;
  for (auto [dim, value] : llvm::enumerate(apply_order(boundary_check_bits))) {
    if (value) {
      new_boundary_check.push_back(dim);
    }
  }
  auto new_load = rewriter.create<LoadOp>(
      load.getLoc(), new_make_ptr, new_boundary_check, load.getPadding(),
      load.getCache(), load.getEvict(), load.getIsVolatile());

  rewriter.replaceOp(op, new_load.getResult());
  return success();
}

LogicalResult PushTransposeUpThroughElementwise(TransOp op,
                                                PatternRewriter& rewriter) {
  Operation* elementwise = op.getSrc().getDefiningOp();
  if (!elementwise || elementwise->getNumResults() != 1 ||
      !elementwise->hasTrait<OpTrait::Elementwise>()) {
    return rewriter.notifyMatchFailure(
        op, "source is not a single-result elementwise op");
  }

  SmallVector<Value> new_operands;
  new_operands.reserve(elementwise->getNumOperands());
  for (Value operand : elementwise->getOperands()) {
    if (auto tensor_type = dyn_cast<RankedTensorType>(operand.getType())) {
      operand = rewriter.create<TransOp>(elementwise->getLoc(), operand,
                                         op.getOrderAttr());
    }
    new_operands.push_back(operand);
  }

  Operation* new_op = rewriter.clone(*elementwise);
  new_op->setOperands(new_operands);
  new_op->getResult(0).setType(op.getType());
  rewriter.replaceOp(op, new_op->getResults());
  return success();
}

SmallVector<int32_t> GetInversePermutation(ArrayRef<int32_t> permutation) {
  SmallVector<int32_t> result(permutation.size());
  for (int32_t i = 0; i < permutation.size(); ++i) {
    result[permutation[i]] = i;
  }
  return result;
}

LogicalResult PushTransposeUpThroughReshape(TransOp op,
                                            PatternRewriter& rewriter) {
  auto reshape = op.getSrc().getDefiningOp<ReshapeOp>();
  if (!reshape) {
    return rewriter.notifyMatchFailure(op,
                                       "Transpose source is not a reshape.");
  }

  auto operand_shape = reshape.getSrc().getType().getShape();
  auto reshape_shape = reshape.getType().getShape();
  SmallVector<std::pair<int64_t, int64_t>> result_to_operand_range(
      reshape_shape.size());

  auto inv_order = GetInversePermutation(op.getOrder());
  auto factors = ::xla::CommonFactors(operand_shape, reshape_shape);
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [operand_from, reshape_from] = factors[i - 1];
    auto [operand_to, reshape_to] = factors[i];

    SmallVector<int64_t> indices;
    indices.reserve(reshape_to - reshape_from);
    for (int64_t j = reshape_from; j < reshape_to; ++j) {
      int32_t index = inv_order[j];
      result_to_operand_range[index] = {operand_from, operand_to};
      operand_from = operand_to;
      indices.push_back(index);
    }

    if (indices.empty() ||
        indices.back() - indices.front() >= reshape_to - reshape_from ||
        !absl::c_is_sorted(indices)) {
      return rewriter.notifyMatchFailure(
          op, "Transposing non-contiguous dimensions.");
    }
  }

  SmallVector<int32_t> new_order;
  new_order.reserve(operand_shape.size());
  for (auto [first, last] : result_to_operand_range) {
    for (int64_t i = first; i < last; ++i) {
      new_order.push_back(i);
    }
  }

  auto new_trans =
      rewriter.create<TransOp>(reshape.getLoc(), reshape.getSrc(), new_order);
  rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), new_trans);
  return success();
}

LogicalResult PushTransposeUpThroughJoinOfInlineAsm(TransOp op,
                                                    PatternRewriter& rewriter) {
  auto join = op.getSrc().getDefiningOp<JoinOp>();
  if (!join) {
    return rewriter.notifyMatchFailure(op, "Transpose source is not a join.");
  }
  if (op.getOrder().back() + 1 != op.getOrder().size()) {
    return rewriter.notifyMatchFailure(op, "Transposes last dimension.");
  }
  auto inline_asm = join.getLhs().getDefiningOp<ElementwiseInlineAsmOp>();
  if (!inline_asm || join.getRhs().getDefiningOp() != inline_asm) {
    return rewriter.notifyMatchFailure(op, "Join source is not an inline asm.");
  }

  SmallVector<Value> new_operands;
  new_operands.reserve(inline_asm->getNumOperands());
  auto order = op.getOrder().drop_back();
  for (Value operand : inline_asm->getOperands()) {
    if (auto tensor_type = dyn_cast<RankedTensorType>(operand.getType())) {
      operand = rewriter.create<TransOp>(inline_asm->getLoc(), operand, order);
    }
    new_operands.push_back(operand);
  }

  Operation* new_inline_asm = rewriter.clone(*inline_asm.getOperation());
  new_inline_asm->setOperands(new_operands);
  for (Value result : new_inline_asm->getResults()) {
    if (auto tensor_type = dyn_cast<RankedTensorType>(result.getType())) {
      auto shape = ApplyPermutation(tensor_type.getShape(), order);
      result.setType(tensor_type.clone(shape));
    }
  }
  rewriter.replaceOpWithNewOp<JoinOp>(op, op.getType(),
                                      new_inline_asm->getResults());

  return success();
}

class TritonXLAFoldTransposePass
    : public impl::TritonXLAFoldTransposePassBase<TritonXLAFoldTransposePass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(FoldTransposeOfLoad);
    patterns.add(PushTransposeUpThroughElementwise);
    patterns.add(PushTransposeUpThroughReshape);
    patterns.add(PushTransposeUpThroughJoinOfInlineAsm);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAFoldTransposePass() {
  return std::make_unique<TritonXLAFoldTransposePass>();
}

}  // namespace mlir::triton::xla

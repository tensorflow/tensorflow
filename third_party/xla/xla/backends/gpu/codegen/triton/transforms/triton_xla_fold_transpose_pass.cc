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
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAFOLDTRANSPOSEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

LogicalResult FoldTransposeOfExtract(TransOp op, PatternRewriter& rewriter) {
  auto extract = op.getSrc().getDefiningOp<ExtractOp>();
  if (!extract) {
    return rewriter.notifyMatchFailure(op, "Transpose source is not extract.");
  }

  auto apply_order = [&](auto range) {
    SmallVector<std::decay_t<decltype(*range.begin())>> result;
    result.reserve(range.size());
    for (int32_t dim : op.getOrder()) {
      result.push_back(range[dim]);
    }
    return result;
  };

  rewriter.replaceOpWithNewOp<ExtractOp>(op, op.getType(), extract.getSrc(),
                                         apply_order(extract.getMixedOffsets()),
                                         apply_order(extract.getMixedStrides()),
                                         apply_order(extract.getSrcShape()),
                                         apply_order(extract.getSrcLayout()));
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

class TritonXLAFoldTransposePass
    : public impl::TritonXLAFoldTransposePassBase<TritonXLAFoldTransposePass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(FoldTransposeOfExtract);
    patterns.add(PushTransposeUpThroughElementwise);
    patterns.add(PushTransposeUpThroughReshape);
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

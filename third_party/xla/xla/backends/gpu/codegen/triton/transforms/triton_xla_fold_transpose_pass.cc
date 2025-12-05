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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAFOLDTRANSPOSEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Sets the insertion point at the given op and returns the guard.
[[nodiscard]] OpBuilder::InsertionGuard SetInsertionPoint(OpBuilder& builder,
                                                          Operation* op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  return guard;
}

// Push the transpose up through the extract tile, this will then be folded into
// MemrefToPtr at the lowering stage.
LogicalResult PushTransposeThroughExtractTile(TransOp op,
                                              PatternRewriter& rewriter) {
  auto extract = op.getSrc().getDefiningOp<::xla::xtile::ExtractTileOp>();
  if (!extract) {
    return rewriter.notifyMatchFailure(op, "Transpose source is not extract.");
  }

  SmallVector<unsigned> reduced_dims =
      to_vector(extract.getReducedDimensions());
  absl::c_sort(reduced_dims);

  // Compute the set of not-reduced dimensions.
  size_t dst_rank = extract.getType().getRank();
  SmallVector<unsigned> retained_dims;
  retained_dims.reserve(dst_rank);
  for (auto [i, dim] : llvm::enumerate(reduced_dims)) {
    for (unsigned j = retained_dims.size() + i; j < dim; ++j) {
      retained_dims.push_back(j);
    }
  }
  while (retained_dims.size() < dst_rank) {
    retained_dims.push_back(retained_dims.size() + reduced_dims.size());
  }

  // Compute the permutation of source dimensions.
  size_t src_rank = extract.getSource().getType().getRank();
  SmallVector<int64_t> permutation;
  permutation.reserve(src_rank);
  for (auto [src_dim, dst_dim] :
       llvm::zip_equal(retained_dims, op.getOrder())) {
    while (permutation.size() < src_dim) {
      permutation.push_back(permutation.size());
    }
    permutation.push_back(retained_dims[dst_dim]);
  }
  while (permutation.size() < src_rank) {
    permutation.push_back(permutation.size());
  }

  auto permute = [&](auto range) {
    SmallVector<std::decay_t<decltype(*range.begin())>> result;
    result.reserve(range.size());
    for (int32_t dim : permutation) {
      result.push_back(range[dim]);
    }
    return result;
  };

  auto permutation_map = mlir::AffineMapAttr::get(
      mlir::AffineMap::getPermutationMap(permutation, rewriter.getContext()));
  // TODO(willfroom): Return a permutation layout (b/455478641).
  auto pushed_transpose = mlir::memref::TransposeOp::create(
      rewriter, extract.getLoc(), extract.getSource(), permutation_map);

  rewriter.replaceOpWithNewOp<::xla::xtile::ExtractTileOp>(
      op, op.getType(), pushed_transpose, permute(extract.getOffsets()),
      permute(extract.getFullTileShape()), permute(extract.getStrides()));

  if (extract->use_empty()) {
    rewriter.eraseOp(extract);
  }

  return success();
}

LogicalResult PushTransposeUpThroughBroadcast(TransOp op,
                                              PatternRewriter& rewriter) {
  auto broadcast = op.getSrc().getDefiningOp<BroadcastOp>();
  if (!broadcast) {
    return rewriter.notifyMatchFailure(  //
        op, "Transpose source is not a broadcast.");
  }
  Value new_trans = TransOp::create(rewriter, op.getLoc(), broadcast.getSrc(),
                                    op.getOrderAttr());
  rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), new_trans);
  return success();
}

LogicalResult PushTransposeUpThroughExpandDims(TransOp op,
                                               PatternRewriter& rewriter) {
  auto expand_dims = op.getSrc().getDefiningOp<ExpandDimsOp>();
  if (!expand_dims) {
    return rewriter.notifyMatchFailure(
        op, "Transpose source is not an expand_dims.");
  }

  unsigned new_axis = [&] {
    for (auto [i, dim] : llvm::enumerate(op.getOrder())) {
      if (dim == expand_dims.getAxis()) {
        return i;
      }
    }
    llvm_unreachable("Transpose order does not contain expand_dims axis");
  }();

  auto new_order = llvm::to_vector(op.getOrder());
  new_order.erase(new_order.begin() + new_axis);
  for (auto& dim : new_order) {
    dim -= dim > expand_dims.getAxis();
  }

  Value new_trans =
      TransOp::create(rewriter, op.getLoc(), expand_dims.getSrc(), new_order);
  rewriter.replaceOpWithNewOp<ExpandDimsOp>(op, op.getType(), new_trans,
                                            new_axis);
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
      operand = TransOp::create(rewriter, elementwise->getLoc(), operand,
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

// Pushes tt.trans up into scf.if.
//
// Example:
//   %0 = scf.if %cond -> type1 {
//     scf.yield %then : type1
//   } else {
//     scf.yield %else : type1
//   }
//   %1 = tt.trans %0 {order = [1, 0]}
// is rewritten to:
//   %0 = scf.if %cond -> type2 {
//     %1 = tt.trans %then {order = [1, 0]}
//     scf.yield %1 : type2
//   } else {
//     %2 = tt.trans %else {order = [1, 0]}
//     scf.yield %2 : type2
//   }
LogicalResult PushTransposeUpIntoIf(TransOp op, PatternRewriter& rewriter) {
  Value src = op.getSrc();
  auto if_op = src.getDefiningOp<scf::IfOp>();
  if (!if_op || !src.hasOneUse()) {
    return rewriter.notifyMatchFailure(op, "Expected scf.if producer.");
  }

  // Compute the new types for the if op.
  unsigned result_number = cast<OpResult>(src).getResultNumber();
  auto new_types = llvm::to_vector(if_op.getResultTypes());
  new_types[result_number] = op.getType();

  auto new_if_op =
      scf::IfOp::create(rewriter, op.getLoc(), new_types, if_op.getCondition(),
                        /*addThenBlock=*/false,
                        /*addElseBlock=*/false);

  // Update then and else regions.
  for (auto [old_region, new_region] :
       llvm::zip(if_op.getRegions(), new_if_op.getRegions())) {
    rewriter.inlineRegionBefore(*old_region, *new_region, new_region->end());
    if (new_region->empty()) {
      continue;
    }
    auto yield_op = new_region->front().getTerminator();
    OpBuilder::InsertionGuard guard = SetInsertionPoint(rewriter, yield_op);
    auto trans_op =
        TransOp::create(rewriter, op.getLoc(), op.getType(),
                        yield_op->getOperand(result_number), op.getOrderAttr());
    yield_op->setOperand(result_number, trans_op);
  }
  rewriter.replaceOp(op, new_if_op.getResult(result_number));
  rewriter.replaceOp(if_op, new_if_op);
  return success();
}

LogicalResult HoistTransposeUpFromIf(TransOp op, PatternRewriter& rewriter) {
  scf::IfOp if_op = dyn_cast<scf::IfOp>(op->getParentOp());
  if (!if_op) {
    return rewriter.notifyMatchFailure(op, "Not a child of scf.if.");
  }
  if (!op.getSrc().getParentRegion()->isAncestor(if_op->getParentRegion())) {
    return rewriter.notifyMatchFailure(op, "Operand defined inside scf.if.");
  }

  op->moveBefore(if_op);
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
      TransOp::create(rewriter, reshape.getLoc(), reshape.getSrc(), new_order);
  rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), new_trans);
  return success();
}

LogicalResult PushTransposeUpThroughMask(TransOp op,
                                         PatternRewriter& rewriter) {
  auto mask_op = op.getSrc().getDefiningOp<::xla::xtile::MaskOp>();
  if (!mask_op) {
    return rewriter.notifyMatchFailure(op, "source is not a mask op");
  }

  llvm::SmallVector<int64_t> new_bounds(op.getOrder().size());
  for (auto [idx, dim] : llvm::enumerate(op.getOrder())) {
    new_bounds[idx] = mask_op.getBounds()[dim];
  }

  auto new_transpose = TransOp::create(rewriter, op.getLoc(),
                                       mask_op.getSource(), op.getOrderAttr());

  rewriter.replaceOpWithNewOp<::xla::xtile::MaskOp>(
      op, op.getType(), new_transpose, new_bounds, mask_op.getValue());
  return success();
}

class TritonXLAFoldTransposePass
    : public impl::TritonXLAFoldTransposePassBase<TritonXLAFoldTransposePass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(PushTransposeThroughExtractTile);
    patterns.add(PushTransposeUpIntoIf);
    patterns.add(HoistTransposeUpFromIf, /*benefit=*/2);
    patterns.add(PushTransposeUpThroughBroadcast);
    patterns.add(PushTransposeUpThroughElementwise);
    patterns.add(PushTransposeUpThroughExpandDims);
    patterns.add(PushTransposeUpThroughReshape);
    patterns.add(PushTransposeUpThroughMask);
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

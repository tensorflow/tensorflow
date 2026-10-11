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

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "triton/Dialect/Triton/IR/Dialect.h"
// Above header needs to be included first to avoid 'major' macro collision.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACOLLAPSECONTIGUOUSMINORDIMSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Return offset_m1 rescaled by the size of m0 dimension.
//
// `minor` is the minormost dimension. Its offset must be 0.
// `major` is the next-to-minormost dimension. Its offset needs to be scaled by
// `minor`'s size.
OpFoldResult ComputeMergedOffset(PatternRewriter& rewriter, Location loc,
                                 OpFoldResult offset_major,
                                 int64_t src_shape_minor) {
  if (std::optional<int64_t> c_offset_major =
          mlir::getConstantIntValue(offset_major);
      c_offset_major.has_value() && *c_offset_major >= 0) {
    return rewriter.getIndexAttr(*c_offset_major * src_shape_minor);
  }

  if (auto val = llvm::dyn_cast_if_present<Value>(offset_major)) {
    if (auto apply_indexing = val.getDefiningOp<::xla::ApplyIndexingOp>()) {
      OpResult op_result = mlir::cast<OpResult>(val);
      unsigned result_idx = op_result.getResultNumber();
      ::xla::IndexingMap indexing_map = apply_indexing.getIndexingMap();
      const ::xla::SymbolicMap& symbolic_map = indexing_map.GetSymbolicMap();

      ::xla::SymbolicExpr scaled_result =
          symbolic_map.GetResult(result_idx) * src_shape_minor;

      ::xla::SymbolicMap new_symbolic_map = ::xla::SymbolicMap::Get(
          rewriter.getContext(), symbolic_map.GetNumDims(),
          symbolic_map.GetNumSymbols(), {scaled_result});
      ::xla::IndexingMap new_indexing_map(
          new_symbolic_map, indexing_map.GetDimVars(),
          indexing_map.GetRangeVars(), indexing_map.GetRTVars(),
          indexing_map.GetSymbolicConstraints());
      new_indexing_map.Simplify();
      ::xla::ApplyIndexingOp new_indexing_op = ::xla::ApplyIndexingOp::create(
          rewriter, loc, apply_indexing.getOperands(), new_indexing_map);
      return new_indexing_op.getResult(0);
    }
  }
  Value val_major =
      getValueOrCreateConstantIndexOp(rewriter, loc, offset_major);
  Value scale = arith::ConstantIndexOp::create(rewriter, loc, src_shape_minor);
  return arith::MulIOp::create(rewriter, loc, val_major, scale).getResult();
}

struct CollapsedOpData {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> layout;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  SmallVector<OpFoldResult> offsets;
};

// If the minor-most dimensions are contiguous in memory, returns the shape with
// minor dimensions collapsed into one. Otherwise, returns nullopt.
//
// Triton doesn't implement emitting vectorized loads/stores that span across
// dimensions. This prevents tensors with small minor-most dimensions from using
// fully vectorized loads/stores. For example, a f32[1024, 2]{1, 0} tensor will
// only use 64-bit wide loads/stores even though memory layout would allow for
// 128-bit ones.
std::optional<CollapsedOpData> CollapseContiguousMinorDims(
    PatternRewriter& rewriter, Location loc, ArrayRef<int64_t> orig_shape,
    ArrayRef<int64_t> orig_layout, ArrayRef<int64_t> orig_sizes,
    ArrayRef<int64_t> orig_strides, ArrayRef<OpFoldResult> orig_offsets,
    RankedTensorType tile_type) {
  Type elem_type = tile_type.getElementType();
  if (!elem_type.isIntOrFloat()) {
    return std::nullopt;
  }

  if (!llvm::isPowerOf2_64(tile_type.getNumElements())) {
    return std::nullopt;
  }
  int64_t bit_width = elem_type.getIntOrFloatBitWidth();

  SmallVector<int64_t> shape = to_vector(orig_shape);
  SmallVector<int64_t> layout = to_vector(orig_layout);
  SmallVector<int64_t> sizes = to_vector(orig_sizes);
  SmallVector<int64_t> strides = to_vector(orig_strides);
  SmallVector<OpFoldResult> offsets = to_vector(orig_offsets);

  bool collapsed_any = false;
  for (int64_t current_rank = shape.size(); current_rank >= 2; --current_rank) {
    // We mutate layout, shape, sizes, strides, and offsets in place, so [0] and
    // [1] are the two minormost dimensions that are not collapsed yet.
    int64_t m0 = layout[0];
    int64_t m1 = layout[1];

    // Simple merge requires m0 to be the most minor logical dim and m1 to be
    // adjacent to it (m1 == m0 - 1). Bail on non-default / non-contiguous
    // layout.
    if (m0 != current_rank - 1 || m1 != m0 - 1) {
      break;
    }

    // Only collapse if doing so allows for wider load/store. SASS has
    // vectorized load/store instructions up to 128 bits.
    static constexpr int64_t kMaxVectorizedOpBitWidth = 128;
    if (sizes[m0] * bit_width >= kMaxVectorizedOpBitWidth) {
      break;
    }

    // We can only collapse if the elements are contiguous in memory.
    // * The minor-most dimension must be fully covered by a single tile.
    if (sizes[m0] != shape[m0]) {
      break;
    }
    // * The offset must be zero.
    if (!mlir::isZeroInteger(offsets[m0])) {
      break;
    }
    // * Strides must be 1.
    if (strides[m0] != 1 || strides[m1] != 1) {
      break;
    }

    // Overflow checks for merged dimension and offset: bail if either exceeds
    // INT32_MAX.
    if (shape[m0] <= 0 || shape[m1] <= 0 ||
        shape[m1] > std::numeric_limits<int32_t>::max() / shape[m0]) {
      break;
    }
    int64_t merged_shape = shape[m1] * shape[m0];
    if (merged_shape > std::numeric_limits<int32_t>::max()) {
      break;
    }
    if (std::optional<int64_t> c1 = mlir::getConstantIntValue(offsets[m1])) {
      if (*c1 < 0 || *c1 > std::numeric_limits<int32_t>::max() / shape[m0]) {
        break;
      }
    }
    if (auto val = llvm::dyn_cast_if_present<Value>(offsets[m1])) {
      if (auto apply_indexing = val.getDefiningOp<::xla::ApplyIndexingOp>()) {
        OpResult op_result = mlir::cast<OpResult>(val);
        unsigned result_idx = op_result.getResultNumber();
        const ::xla::IndexingMap& indexing_map =
            apply_indexing.getIndexingMap();
        ::xla::Interval range =
            indexing_map.GetRangeEvaluator().ComputeExpressionRange(
                indexing_map.GetSymbolicMap().GetResult(result_idx));
        if (range.lower < 0 ||
            range.upper > std::numeric_limits<int32_t>::max() / shape[m0]) {
          break;
        }
      }
    }

    // Collapse m1 and m0.
    int64_t merged_size = sizes[m1] * sizes[m0];
    OpFoldResult merged_offset =
        ComputeMergedOffset(rewriter, loc, offsets[m1], shape[m0]);

    shape.pop_back();
    shape.back() = merged_shape;

    sizes.pop_back();
    sizes.back() = merged_size;

    strides.pop_back();
    // We already checked that strides.back() is 1.

    offsets.pop_back();
    offsets.back() = merged_offset;

    layout.erase(layout.begin());

    collapsed_any = true;
  }

  if (!collapsed_any) {
    return std::nullopt;
  }

  return CollapsedOpData{
      std::move(shape),   std::move(layout),  std::move(sizes),
      std::move(strides), std::move(offsets),
  };
}

// Computes the tensor shape for the collapsed extract/insert op, preserving
// rank-reduced unit dimensions from the original op.
SmallVector<int64_t> GetCollapsedTensorShape(
    ArrayRef<int64_t> orig_sizes, ArrayRef<int64_t> collapsed_sizes,
    ArrayRef<int64_t> orig_tensor_shape) {
  std::optional<llvm::SmallDenseSet<unsigned>> reduced_dims =
      mlir::computeRankReductionMask(orig_sizes, orig_tensor_shape);
  if (!reduced_dims.has_value() || reduced_dims->empty()) {
    return to_vector(collapsed_sizes);
  }

  SmallVector<int64_t> tensor_shape;
  int64_t first_collapsed_dim =
      static_cast<int64_t>(collapsed_sizes.size()) - 1;
  for (int64_t d = 0; d < first_collapsed_dim; ++d) {
    if (!reduced_dims->contains(d)) {
      tensor_shape.push_back(collapsed_sizes[d]);
    }
  }

  bool all_collapsed_dims_reduced = true;
  for (int64_t d = first_collapsed_dim;
       d < static_cast<int64_t>(orig_sizes.size()); ++d) {
    if (!reduced_dims->contains(d)) {
      all_collapsed_dims_reduced = false;
      break;
    }
  }

  if (!all_collapsed_dims_reduced || tensor_shape.empty()) {
    tensor_shape.push_back(collapsed_sizes.back());
  }
  return tensor_shape;
}

class CollapseExtractOp : public mlir::OpRewritePattern<ExtractOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter& rewriter) const override {
    std::optional<CollapsedOpData> collapsed_data = CollapseContiguousMinorDims(
        rewriter, op.getLoc(), op.getSrcShape(), op.getSrcLayout(),
        op.getStaticSizes(), op.getStaticStrides(), op.getMixedOffsets(),
        op.getType());
    if (!collapsed_data.has_value()) {
      return failure();
    }

    SmallVector<int64_t> collapsed_tensor_shape = GetCollapsedTensorShape(
        op.getStaticSizes(), collapsed_data->sizes, op.getType().getShape());
    RankedTensorType collapsed_type = RankedTensorType::get(
        collapsed_tensor_shape, op.getType().getElementType());

    ExtractOp new_extract = ExtractOp::create(
        rewriter, op.getLoc(), collapsed_type, op.getSrc(),
        collapsed_data->offsets, collapsed_data->sizes, collapsed_data->strides,
        collapsed_data->shape, collapsed_data->layout);
    new_extract->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    Value result = new_extract.getResult();
    if (collapsed_type != op.getType()) {
      result = triton::ReshapeOp::create(rewriter, op.getLoc(),
                                         op.getType().getShape(), result,
                                         /*allowReorder=*/false);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class CollapseInsertOp : public mlir::OpRewritePattern<InsertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    RankedTensorType src_type =
        mlir::cast<RankedTensorType>(op.getSrc().getType());
    std::optional<CollapsedOpData> collapsed_data = CollapseContiguousMinorDims(
        rewriter, op.getLoc(), op.getDstShape(), op.getDstLayout(),
        op.getStaticSizes(), op.getStaticStrides(), op.getMixedOffsets(),
        src_type);
    if (!collapsed_data.has_value()) {
      return failure();
    }

    SmallVector<int64_t> collapsed_tensor_shape = GetCollapsedTensorShape(
        op.getStaticSizes(), collapsed_data->sizes, src_type.getShape());
    Value new_src = op.getSrc();
    if (ArrayRef<int64_t>(collapsed_tensor_shape) != src_type.getShape()) {
      new_src = triton::ReshapeOp::create(rewriter, op.getLoc(),
                                          collapsed_tensor_shape, new_src,
                                          /*allowReorder=*/false);
    }

    InsertOp new_insert = InsertOp::create(
        rewriter, op.getLoc(), new_src, op.getDst(), collapsed_data->offsets,
        collapsed_data->sizes, collapsed_data->strides, collapsed_data->shape,
        collapsed_data->layout);
    new_insert->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    rewriter.eraseOp(op);
    return success();
  }
};

class TritonXLACollapseContiguousMinorDimsPass
    : public impl::TritonXLACollapseContiguousMinorDimsPassBase<
          TritonXLACollapseContiguousMinorDimsPass> {
 public:
  using TritonXLACollapseContiguousMinorDimsPassBase::
      TritonXLACollapseContiguousMinorDimsPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<CollapseExtractOp, CollapseInsertOp>(ctx);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla

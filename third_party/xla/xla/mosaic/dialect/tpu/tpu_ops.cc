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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"
#include "xla/layout.h"

namespace mlir {
namespace tpu {

namespace {

llvm::RoundingMode convertTpuRoundingModeToLLVMIR(tpu::RoundingMode mode) {
  switch (mode) {
    case tpu::RoundingMode::kToNearestEven:
      return llvm::RoundingMode::NearestTiesToEven;
    case tpu::RoundingMode::kTowardsZero:
      return llvm::RoundingMode::TowardZero;
  }
}

// Attempts to convert `sourceValue` to an APFloat value with
// `targetSemantics` and `roundingMode`, without any information loss.
static FailureOr<APFloat> convertFloatValue(
    APFloat sourceValue, const llvm::fltSemantics& targetSemantics,
    llvm::RoundingMode roundingMode = llvm::RoundingMode::NearestTiesToEven) {
  bool losesInfo = false;
  auto status = sourceValue.convert(targetSemantics, roundingMode, &losesInfo);
  if (losesInfo || status != APFloat::opOK) {
    return failure();
  }

  return sourceValue;
}

std::optional<CoreType> getRefCoreType(TypedValue<MemRefType> value) {
  auto space = dyn_cast_if_present<tpu::MemorySpaceAttr>(
      value.getType().getMemorySpace());
  if (!space) {
    return std::nullopt;
  }
  return space.getCoreType();
}

template <typename OpTy>
LogicalResult verifyPackOp(OpTy op, int32_t max_size) {
  if (op.getSources().empty()) {
    return op.emitOpError("At least one source is required");
  }
  if (!llvm::all_of(op.getSources(), [&](Value source) {
        return source.getType() == op.getSources().front().getType();
      })) {
    return op.emitOpError("All sources must have the same type");
  }
  if (op.getPositions().size() != op.getSources().size()) {
    return op.emitOpError("Size of sources and positions must match");
  }
  if (op.getSources().size() > max_size) {
    return op.emitOpError("Number of sources must be less than max_size (")
           << max_size << "), got " << op.getSources().size();
  }
  SmallVector<bool> seen_positions(max_size, false);
  for (const int32_t position : op.getPositions()) {
    if (position < 0 || max_size <= position) {
      return op.emitOpError("Positions must be between 0 and max_size (")
             << max_size << "), got " << position;
    }
    if (seen_positions[position]) {
      return op.emitOpError("Positions must be unique");
    }
    seen_positions[position] = true;
  }
  return success();
}

}  // namespace

LogicalResult UnrollVectorsOp::canonicalize(UnrollVectorsOp op,
                                            PatternRewriter& rewriter) {
  RollVectorsOp roll_op = op.getOperand().getDefiningOp<RollVectorsOp>();
  if (!roll_op) {
    return failure();
  }
  if (roll_op.getNumOperands() != op.getNumResults()) {
    return failure();
  }
  for (auto [v1, v2] :
       llvm::zip(roll_op.getOperandTypes(), op.getResultTypes())) {
    if (v1 != v2) {
      return failure();
    }
  }
  rewriter.replaceOp(op, roll_op.getOperands());
  return success();
}

LogicalResult BitcastOp::verify() {
  auto in_ty = getInput().getType();
  auto out_ty = getOutput().getType();
  auto in_bitwidth = getElementTypeBitwidth(in_ty);
  auto out_bitwidth = getElementTypeBitwidth(out_ty);
  if (in_bitwidth != out_bitwidth) {
    if (in_ty.getRank() < 2 || out_ty.getRank() < 2) {
      return emitError(
          "Not implemented: bitcast between different bitwidths on a 1D "
          "vector.");
    }
    SmallVector<int64_t, 4> in_shape(in_ty.getShape());
    SmallVector<int64_t, 4> out_shape(out_ty.getShape());
    *(in_shape.end() - 2) *= in_bitwidth;
    *(out_shape.end() - 2) *= out_bitwidth;
    if (in_shape != out_shape) {
      return emitError(
          "Expected input and output shapes are the same after multiplying the "
          "second-minor dimension by the ratio of bitwidths.");
    }
  } else if (in_ty.getShape() != out_ty.getShape()) {
    return emitError(
        "Expected input and output shapes are the same when bitwidth does not "
        "change.");
  }
  return success();
}

OpFoldResult BitcastVregOp::fold(FoldAdaptor adaptor) {
  // Bitcast from X -> X is a no-op.
  if (getType() == getInput().getType()) {
    return getInput();
  }
  // Simplify bitcast chain of X -> Y -> Z to X -> Z.
  if (auto defining_op = getInput().getDefiningOp<BitcastVregOp>()) {
    getInputMutable().assign(defining_op.getInput());
    return getResult();
  }
  return nullptr;
}

LogicalResult MemRefSliceOp::verify() {
  auto source_type = getMemRef().getType();
  auto target_type = getType();
  auto source_layout = source_type.getLayout();
  auto target_layout = target_type.getLayout();
  auto target_memory_space = target_type.getMemorySpace();
  auto indices = getBaseIdx();
  auto slice_shape = getResult().getType().getShape();
  if (!source_type.hasStaticShape()) {
    return emitOpError(
        "Only slicing of memrefs with static shapes is supported.");
  }
  if (getDynamicSizes().size() != target_type.getNumDynamicDims()) {
    return emitOpError(
        "Number of provided dynamic dimensions sizes must match the number of "
        "dynamic dimensions in the target type.");
  }
  auto source_shape = source_type.getShape();
  bool is_semaphore =
      HasMemorySpace(source_type, tpu::MemorySpace::kSemaphoreMem);
  if (is_semaphore &&
      !isa<SemaphoreType, DMASemaphoreType>(source_type.getElementType())) {
    return emitOpError(
        "References to semaphore memory space must have a semaphore element "
        "type.");
  }
  if (indices.size() != slice_shape.size() ||
      indices.size() != source_shape.size()) {
    return emitOpError("Indices and slice shapes must match.");
  }
  // TODO(apaszke): Check that the result has a smaller shape.
  // TODO(apaszke): Check that strides are equivalent.
  // Source and target memory spaces may be different before propagation is done
  // by memory space specialization.
  bool is_target_memory_space_provided = target_memory_space != nullptr;
  if (is_target_memory_space_provided &&
      target_memory_space != source_type.getMemorySpace()) {
    return emitOpError(
        "Memory spaces must match if the target memory space is provided.");
  }
  if (isa<TiledLayoutAttr>(source_layout) !=
      isa<TiledLayoutAttr>(target_layout)) {
    return emitOpError("Source and target layouts must match.");
  }
  auto tiled_layout = dyn_cast<TiledLayoutAttr>(source_layout);
  if (!tiled_layout) {
    return success();
  }
  ArrayRef<xla::Tile> tiles = tiled_layout.getTiles();
  if (tiles.empty()) {
    return success();
  }
  const xla::Tile& first_tile = tiles.front();
  const int64_t tile_rank = first_tile.dimensions().size();
  const int64_t rank = source_shape.size();
  if (rank < tile_rank) {
    return emitOpError("Source memref has ")
           << rank << " dimensions, but the tiling has " << tile_rank
           << " dimensions. The memref must have at least as many dimensions "
              "as the tiling.";
  }
  return success();
}

mlir::InFlightDiagnostic MemRefSliceOp::verifyOffsetAndSizeTileAlignment(
    std::array<int64_t, 2> tc_target_shape, MemRefType source_type_override) {
  mlir::MemRefType source_ty = getMemRef().getType();
  if (source_type_override) {
    source_ty = source_type_override;
  }
  auto tiled_layout = cast<TiledLayoutAttr>(source_ty.getLayout());
  if (getResult().getType().getLayout() != tiled_layout) {
    return emitOpError("Operand and result layouts must be equal.");
  }
  ArrayRef<xla::Tile> tiles = tiled_layout.getTiles();
  if (tiles.empty()) {
    return {};
  }
  const xla::Tile& first_tile = tiles.front();
  const int64_t tile_rank = first_tile.dimensions().size();
  const int64_t untiled_dims = source_ty.getRank() - tile_rank;
  MemRefType result_type = getResult().getType();
  int64_t dynamic_dim_idx = llvm::count(
      result_type.getShape().take_front(untiled_dims), ShapedType::kDynamic);
  const int64_t sublane_count = tc_target_shape[0];
  const int64_t lane_count = tc_target_shape[1];
  for (int64_t i = 0; i < tile_rank; ++i) {
    int64_t tile_dim = first_tile.dimension(i);
    const int64_t dim = untiled_dims + i;
    if (!isGuaranteedDivisible(getBaseIdx()[dim], tile_dim)) {
      return emitOpError(
                 "Offsets along tiled dimensions must be aligned to "
                 "tiles. Failed to verify that the index at dimension ")
             << dim << " is divisible by the tile dimension " << tile_dim
             << ". If it is, use tpu.assume_multiple to suppress this "
                "error.";
    }
    // We only require alignment to compact 2nd minor for large 2nd minor.
    if (tile_rank == 2 && i == 0) {
      int64_t packing = tile_dim / sublane_count;
      if (tile_dim == sublane_count * packing &&
          first_tile.dimension(1) == lane_count && packing > 1 &&
          packing <= sublane_count) {
        tile_dim = sublane_count;
      }
    }
    bool size_is_aligned;
    if (result_type.getShape()[dim] == ShapedType::kDynamic) {
      size_is_aligned =
          isGuaranteedDivisible(getDynamicSizes()[dynamic_dim_idx++], tile_dim);
    } else if (result_type.getShape()[dim] == source_ty.getShape()[dim]) {
      // If the result shape is the same as the source shape, we don't put
      // any restrictions here. The only way to get such a ref is from a kernel
      // input (as any prior slice would be rejected), which implies the data
      // is padded to tiling.
      size_is_aligned = true;
    } else {
      size_is_aligned = result_type.getShape()[dim] % tile_dim == 0;
    }
    if (!size_is_aligned) {
      return emitOpError(
                 "Slice sizes along tiled dimensions must be aligned to "
                 "tiles. Failed to verify that the size at dimension ")
             << dim << " is divisible by the tile dimension " << tile_dim
             << ". If it is, use tpu.assume_multiple to suppress this "
                "error.";
    }
  }
  return {};
}

struct MemRefSliceFoldConstantDynamicDim
    : public OpRewritePattern<MemRefSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefSliceOp op,
                                PatternRewriter& rewriter) const override {
    if (llvm::none_of(op.getDynamicSizes(), [](Value dynamic_size) {
          APInt constant_value;  // Would be nice if we could pass nullptr below
          return matchPattern(dynamic_size, m_ConstantInt(&constant_value));
        })) {
      return failure();
    }
    SmallVector<int64_t> new_shape(op.getType().getShape());
    SmallVector<Value> new_dynamic_sizes;
    int64_t dynamic_dim_index = 0;
    for (Value dynamic_size : op.getDynamicSizes()) {
      // Find the index of the corresponding dynamic dimension in the shape
      while (new_shape[dynamic_dim_index] != ShapedType::kDynamic) {
        ++dynamic_dim_index;
        CHECK(dynamic_dim_index < new_shape.size());
      }
      APInt constant_value;
      if (matchPattern(dynamic_size, m_ConstantInt(&constant_value))) {
        if (constant_value.getSExtValue() <= 0) {
          return op.emitWarning() << "Non-positive constant for dynamic size";
        }
        new_shape[dynamic_dim_index] = constant_value.getSExtValue();
      } else {
        new_dynamic_sizes.push_back(dynamic_size);
      }
      ++dynamic_dim_index;
    }
    // Update the memref_slice op and create a cast op to convert to the old
    // type.
    MemRefType old_type = op.getType();
    MemRefType new_type = MemRefType::Builder(old_type).setShape(new_shape);
    rewriter.modifyOpInPlace(op, [&]() {
      op.getResult().setType(new_type);
      op.getDynamicSizesMutable().assign(new_dynamic_sizes);
    });
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto cast_op = memref::CastOp::create(rewriter, op.getLoc(), old_type, op);
    rewriter.replaceAllUsesExcept(op, cast_op, cast_op);
    return success();
  }
};

OpFoldResult MemRefSliceOp::fold(FoldAdaptor adaptor) {
  if (!getDynamicSizes().empty()) {
    return nullptr;
  }
  if (getMemRef().getType() != getType()) {
    return nullptr;
  }
  for (Value idx : getBaseIdx()) {
    APInt constant_value;
    if (!matchPattern(idx, m_ConstantInt(&constant_value)) ||
        constant_value != 0) {
      return nullptr;
    }
  }
  return getMemRef();
}

void MemRefSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<MemRefSliceFoldConstantDynamicDim>(context);
}

LogicalResult MemRefSqueezeOp::verify() {
  MemRefType source_type = getInput().getType();
  MemRefType target_type = getType();

  if (target_type.getMemorySpace() != nullptr &&
      target_type.getMemorySpace() != source_type.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }

  if (target_type.getElementType() != source_type.getElementType()) {
    return emitOpError("Element types don't match.");
  }

  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  FAILUREOR_ASSIGN_OR_RETURN(
      auto squeezed,
      computeSqueezedDimsChecked(*this, source_shape, target_shape));
  if (squeezed.empty() && source_shape != target_shape) {
    return emitOpError(
        "Source and target shapes must be the same if no dimensions are "
        "squeezed.");
  }

  auto source_layout = source_type.getLayout();
  auto target_layout = target_type.getLayout();
  bool has_tiled_layout = isa<TiledLayoutAttr>(source_layout);
  if (has_tiled_layout != isa<TiledLayoutAttr>(target_layout)) {
    return emitOpError(
        "Either both src and dst or none of them should have a tiled layout");
  }
  if (has_tiled_layout) {
    return verifyTiling();
  }
  return success();
}

mlir::InFlightDiagnostic MemRefSqueezeOp::verifyTiling() {
  MemRefType source_type = getInput().getType();
  auto source_shape = source_type.getShape();
  auto target_shape = getType().getShape();
  auto squeezed_or =
      computeSqueezedDimsChecked(*this, source_shape, target_shape);
  if (failed(squeezed_or)) {
    return {};
  }
  auto& squeezed = squeezed_or.value();

  auto tiles = cast<TiledLayoutAttr>(source_type.getLayout()).getTiles();
  switch (tiles.size()) {
    case 0:
      break;
    case 1: {
      auto tile = tiles.front();
      auto tile_dims = tile.dimensions();
      int first_tiled = source_shape.size() - tile_dims.size();
      for (int dim : squeezed) {
        if (dim >= first_tiled) {
          int tile_idx = dim - first_tiled;
          if (tile_idx < 0 || tile_idx >= static_cast<int>(tile_dims.size())) {
            return emitOpError() << "Internal error: tile index out of bounds.";
          }
          if (tile_dims[tile_idx] != 1) {
            return emitOpError()
                   << "All tiled squeezed dimensions must be of size 1.";
          }
        }
      }
      break;
    }
    default: {
      auto first_tile = tiles.front();
      for (int dim : squeezed) {
        int first_tiled = source_shape.size() - first_tile.dimensions().size();
        if (dim >= first_tiled) {
          return emitOpError() << "When multiple tiles are present, no tiled "
                                  "dimensions can be squeezed.";
        }
      }
    }
  }

  return {};
}

// Rewrites
//
//   tpu.memref_squeeze(memref.cast(x))
//
// to
//
//   memref.cast(tpu.memref_squeeze(x))
//
// when the cast widens static dimensions to dynamic.
struct MemRefSqueezeFoldCast : public OpRewritePattern<MemRefSqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefSqueezeOp op,
                                PatternRewriter& rewriter) const override {
    auto cast_op = op.getInput().getDefiningOp<memref::CastOp>();
    if (!cast_op) {
      return failure();
    }

    TypedValue<BaseMemRefType> cast_source = cast_op.getSource();
    auto cast_source_type = cast<MemRefType>(cast_source.getType());
    auto cast_result_type = cast<MemRefType>(cast_op.getType());
    if (cast_source_type.getRank() != cast_result_type.getRank()) {
      return failure();
    }
    for (auto [source_dim, result_dim] :
         llvm::zip(cast_source_type.getShape(), cast_result_type.getShape())) {
      if (source_dim == result_dim) continue;
      if (ShapedType::isDynamic(source_dim) &&
          !ShapedType::isDynamic(result_dim)) {
        // The result type must be more dynamic than the source type.
        return failure();
      }
    }

    MemRefType result_type = op.getType();
    FAILUREOR_ASSIGN_OR_RETURN(
        SmallVector<int> squeezed,
        computeSqueezedDimsChecked(op, cast_result_type.getShape(),
                                   result_type.getShape()));

    SmallVector<int64_t> new_result_shape;
    for (auto [i, dim] : llvm::enumerate(cast_source_type.getShape())) {
      if (!absl::c_contains(squeezed, i)) {
        new_result_shape.push_back(dim);
      }
    }
    CHECK_EQ(new_result_shape.size(), result_type.getRank());
    if (result_type.getShape() == ArrayRef<int64_t>(new_result_shape)) {
      return failure();
    }

    MemRefType new_result_type =
        MemRefType::Builder(result_type).setShape(new_result_shape);
    auto new_squeeze = MemRefSqueezeOp::create(rewriter, op.getLoc(),
                                               new_result_type, cast_source);
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, result_type, new_squeeze);
    return success();
  }
};

void MemRefSqueezeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.add<MemRefSqueezeFoldCast>(context);
}

LogicalResult RelayoutOp::verify() {
  auto in_layout_array_attr =
      getOperation()->getAttrOfType<ArrayAttr>("in_layout");
  if (!in_layout_array_attr) {
    return emitOpError("missing 'in_layout' attribute");
  }
  if (in_layout_array_attr.size() != 1) {
    return emitOpError(
        "'in_layout' attribute must be an array containing a single "
        "VectorLayoutAttr");
  }
  if (!isa<tpu::VectorLayoutAttr>(in_layout_array_attr[0])) {
    return emitOpError("'in_layout' attribute is not a VectorLayoutAttr");
  }

  auto out_layout_array_attr =
      getOperation()->getAttrOfType<ArrayAttr>("out_layout");
  if (!out_layout_array_attr) {
    return emitOpError("missing 'out_layout' attribute");
  }
  if (out_layout_array_attr.size() != 1) {
    return emitOpError(
        "'out_layout' attribute must be an array containing a single "
        "VectorLayoutAttr");
  }
  if (!isa<tpu::VectorLayoutAttr>(out_layout_array_attr[0])) {
    return emitOpError("'out_layout' attribute is not a VectorLayoutAttr");
  }
  return success();
}

LogicalResult MemRefReshapeOp::verify() {
  auto src_ty = getInput().getType();
  auto tgt_ty = getType();
  if (tgt_ty.getMemorySpace() != nullptr &&
      tgt_ty.getMemorySpace() != src_ty.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }
  if (src_ty.getShape().size() < 2 || tgt_ty.getShape().size() < 2) {
    return emitError("Not implemented: 1d memref reshape.");
  }
  if (tgt_ty.getElementType() != src_ty.getElementType()) {
    return emitOpError("Element types don't match.");
  }
  auto src_elements_num = ShapedType::getNumElements(src_ty.getShape());
  auto tgt_elements_num = ShapedType::getNumElements(tgt_ty.getShape());
  if (src_elements_num != tgt_elements_num) {
    return emitOpError(
        "Number of elements doesn't match between input and output memref "
        "type.");
  }
  if (src_ty.getLayout().isIdentity() && tgt_ty.getLayout().isIdentity()) {
    return success();  // Untiled reshape
  }
  auto src_layout = dyn_cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  auto tgt_layout = dyn_cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (!src_layout || !tgt_layout) {
    return emitOpError("Only identity or tiled layouts supported.");
  }
  return verifyTiling();
}

mlir::InFlightDiagnostic MemRefReshapeOp::verifyTiling() {
  auto src_ty = getInput().getType();
  auto tgt_ty = getType();
  auto src_layout = cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  auto tgt_layout = cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (src_layout.getTiles() != tgt_layout.getTiles()) {
    return emitOpError(
        "Expected the same tiling for the input and output memref.");
  }
  if (src_layout.getTiles().empty()) {
    return {};  // Untiled reshape
  }
  auto tile = src_layout.getTiles().front().dimensions();
  if (tile.size() != 2) {
    return emitOpError("Not implemented: memref reshape with 1D tiling.");
  }
  if (!src_layout.tilesAreKnownContiguous(src_ty.getShape()) ||
      !tgt_layout.tilesAreKnownContiguous(tgt_ty.getShape())) {
    return emitOpError("Not implemented: reshape on a non-contiguous memref.");
  }
  auto src_tiled_shape = src_ty.getShape().take_back(tile.size());
  auto tgt_tiled_shape = tgt_ty.getShape().take_back(tile.size());
  bool is_src_align_tile_2nd_minor = src_tiled_shape[0] % tile[0] == 0;
  bool is_src_align_tile_minor = src_tiled_shape[1] % tile[1] == 0;
  bool is_tgt_align_tile_2nd_minor = tgt_tiled_shape[0] % tile[0] == 0;
  bool is_tgt_align_tile_minor = tgt_tiled_shape[1] % tile[1] == 0;
  if (tile[0] == 1 && is_src_align_tile_minor && is_tgt_align_tile_minor) {
    // When the tiling is (1, ?) and the source and target shapes are aligned
    // to the tile, we support reshape on any dims.
  } else if (tgt_tiled_shape[1] != src_tiled_shape[1]) {
    return emitOpError("Expected the minormost dimension to be unchanged");
  } else if (tgt_tiled_shape[0] != src_tiled_shape[0]) {
    if (!is_src_align_tile_2nd_minor || !is_tgt_align_tile_2nd_minor) {
      return emitOpError(
          "Expected the 2nd minor dimension is aligned to the tile");
    }
  }
  return {};
}

LogicalResult TransposeOp::verify() {
  auto source_type = getSourceVectorType();
  auto permutation = getPermutation();
  auto output_type = getResultVectorType();
  auto input_shape = source_type.getShape();
  auto output_shape = output_type.getShape();
  if (source_type.getElementType() != output_type.getElementType()) {
    return emitOpError("Expected input and output element types to match");
  }
  if (permutation.size() != source_type.getRank()) {
    return emitOpError("Expected permutation rank to match input rank");
  }
  if (permutation.size() != output_type.getRank()) {
    return emitOpError("Expected permutation rank to match output rank");
  }
  std::vector<bool> seen_dims(source_type.getRank(), false);
  for (int64_t dim : permutation) {
    if (dim < 0 || dim >= source_type.getRank()) {
      return emitOpError("Permutation element out of bounds: ") << dim;
    }
    if (seen_dims[dim]) {
      return emitOpError("Permutation element repeated: ") << dim;
    }
    seen_dims[dim] = true;
  }
  for (int i = 0; i < source_type.getRank(); ++i) {
    if (input_shape[permutation[i]] != output_shape[i]) {
      return emitOpError(
          "Expected input shape permuted by the given permutation to match the "
          "output shape");
    }
  }
  return success();
}

LogicalResult MemRefBitcastOp::verify() {
  auto src_ty = getMemRefType(getInput());
  auto tgt_ty = getType();
  if (tgt_ty.getMemorySpace() != nullptr &&
      tgt_ty.getMemorySpace() != src_ty.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }
  if (src_ty.getRank() != tgt_ty.getRank()) {
    return emitOpError("Ranks do not match.");
  }
  if (src_ty.getRank() <= 1) {
    return emitOpError("Not implemented: 1d memref bitcast.");
  }
  auto src_bitwidth = getElementTypeBitwidth(src_ty);
  auto tgt_bitwidth = getElementTypeBitwidth(tgt_ty);
  for (int i = 0; i < src_ty.getRank(); ++i) {
    auto src_dim_size = src_ty.getDimSize(i);
    auto tgt_dim_size = tgt_ty.getDimSize(i);
    if (i == src_ty.getRank() - 2) {
      auto src_bits = src_dim_size * src_bitwidth;
      auto tgt_bits = tgt_dim_size * tgt_bitwidth;
      if (src_bits != tgt_bits) {
        return emitOpError(
                   "Expected the same number of bits on the 2nd minormost "
                   "dim: (")
               << src_dim_size << " * " << src_bitwidth << ") vs ("
               << tgt_dim_size << " * " << tgt_bitwidth << ")";
        ;
      }
    } else {
      if (src_dim_size != tgt_dim_size) {
        return emitOpError("Expected the same dim size on dim ")
               << i << ": " << src_dim_size << " vs " << tgt_dim_size;
      }
    }
  }
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  auto tgt_layout = dyn_cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (!tgt_layout) {
    return success();
  }
  auto src_layout = dyn_cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  if (!src_layout) {
    return emitOpError("Expected a tiled layout for the input memref.");
  }
  return verifyTiling();
}

mlir::InFlightDiagnostic MemRefBitcastOp::verifyTiling() {
  auto src_ty = getMemRefType(getInput());
  auto tgt_ty = getType();
  auto src_bitwidth = getElementTypeBitwidth(src_ty);
  auto tgt_bitwidth = getElementTypeBitwidth(tgt_ty);
  auto src_layout = cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  auto tgt_layout = cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  // TODO(jevinjiang): verify memref tiling is valid. Here we just assume the
  // source and target tilings are valid.
  auto src_tile = src_layout.getTiles().front().dimensions();
  auto tgt_tile = tgt_layout.getTiles().front().dimensions();
  if (src_tile[0] * src_bitwidth != tgt_tile[0] * tgt_bitwidth) {
    return emitOpError("Invalid memref bitcast.");
  }
  return {};
}

LogicalResult MemRefBitcastOp::canonicalize(MemRefBitcastOp op,
                                            PatternRewriter& rewriter) {
  auto src_ty = op.getInput().getType();
  auto dst_ty = op.getType();
  if (src_ty == dst_ty) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  return failure();
}

template <typename Op>
LogicalResult verifyStridedOp(Op op, MemRefType memref_ty, VectorType vector_ty,
                              int64_t min_stride) {
  auto indices = op.getIndices();
  auto strides = op.getStrides();
  if (memref_ty.getRank() != indices.size()) {
    op.emitError("Base memref's rank and indices size do not match: ")
        << memref_ty.getRank() << " vs " << indices.size();
    return failure();
  }
  if (memref_ty.getRank() != strides.size()) {
    op.emitError("Base memref's rank and strides size do not match: ")
        << memref_ty.getRank() << " vs " << strides.size();
    return failure();
  }
  if (memref_ty.getRank() != vector_ty.getRank()) {
    op.emitError("Base memref's rank and result's rank do not match: ")
        << memref_ty.getRank() << " vs " << vector_ty.getRank();
    return failure();
  }
  for (int64_t i = 0; i < memref_ty.getRank(); ++i) {
    if (strides[i] < min_stride) {
      op.emitError("Strides[")
          << i << "]=" << strides[i] << " must be >= " << min_stride;
      return failure();
    }
  }
  return success();
}

LogicalResult StridedLoadOp::verify() {
  return verifyStridedOp<StridedLoadOp>(*this, getBase().getType(), getType(),
                                        /*min_stride=*/0);
}

LogicalResult StridedStoreOp::verify() {
  return verifyStridedOp<StridedStoreOp>(*this, getBase().getType(),
                                         getValueToStore().getType(),
                                         /*min_stride=*/1);
}

template <typename Op>
LogicalResult verifyStoreOp(Op op) {
  MemRefType ref_ty = op.getBase().getType();
  if (ref_ty.getMemorySpace() && !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
    return op.emitOpError("Expected base memref to be in VMEM.");
  }
  VectorType value_ty = op.getValueToStore().getType();
  if (value_ty.getElementType() != ref_ty.getElementType()) {
    return op.emitOpError(
        "Expected base and valueToStore element type to match");
  }
  if (op.getMask()) {
    if (getElementTypeBitwidth(value_ty) != 32) {
      return op.emitError(
          "Not implemented: masked store with non-32-bit element type");
    }
    if (value_ty.getShape() != op.getMask().getType().getShape())
      return op.emitOpError("Expected mask shape to match result shape: (")
             << value_ty.getShape() << "). Got: ("
             << op.getMask().getType().getShape() << ").";
  }
  return success();
}

LogicalResult VectorStoreOp::verify() {
  if (!getStrides().empty()) {
    return emitError("Not implemented: general vector store with strides.");
  }
  MemRefType ref_ty = getBase().getType();
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError("Expected ") << ref_ty.getRank() << " indices.";
  }
  return verifyStoreOp(*this);
}

void VectorStoreOp::build(OpBuilder& builder, OperationState& state,
                          Value valueToStore, Value base, ValueRange indices,
                          Value mask, bool add) {
  build(builder, state, valueToStore, base, indices,
        /*strides=*/builder.getDenseI32ArrayAttr({}), mask, add);
}

template <typename Op>
LogicalResult verifyLoadOp(Op op) {
  MemRefType ref_ty = op.getBase().getType();
  if (ref_ty.getMemorySpace() && !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
    return op.emitOpError("Expected base memref to be in VMEM.");
  }
  VectorType value_ty = op.getResult().getType();
  if (value_ty.getElementType() != ref_ty.getElementType()) {
    return op.emitOpError("Expected base and result element type to match.");
  }
  if (op.getMask()) {
    if (getElementTypeBitwidth(value_ty) != 32) {
      return op.emitError(
          "Not implemented: masked load with non-32-bit element type");
    }
    if (vector::isBroadcastableTo(op.getMask().getType(), value_ty) !=
        vector::BroadcastableToResult::Success) {
      return op.emitOpError(
          "Expected mask shape to be broadcastable to result shape.");
    }
  }
  return success();
}

LogicalResult VectorLoadOp::verify() {
  const MemRefType ref_ty = getBase().getType();
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError("Expected ") << ref_ty.getRank() << " indices.";
  }
  if (!getStrides().empty()) {
    if (llvm::size(getStrides()) != ref_ty.getRank()) {
      return emitOpError("Expected ") << ref_ty.getRank() << " strides.";
    }
    return emitError("Not implemented: general vector load with strides.");
  }
  return verifyLoadOp(*this);
}

void VectorLoadOp::build(OpBuilder& builder, OperationState& state,
                         Type result_type, Value base, ValueRange indices,
                         Value mask) {
  build(builder, state, result_type, base, indices,
        /*strides=*/builder.getDenseI32ArrayAttr({}), mask);
}

LogicalResult VectorLoadIdxOp::verify() {
  VectorType value_ty = getResult().getType();
  MemRefType ref_ty = getBase().getType();
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError(
               "Expected one index vector for each dimension of the base "
               "memref with dimension: ")
           << ref_ty.getRank() << ". Got: " << llvm::size(getIndices()) << ".";
  }
  for (const auto [i, index] : llvm::enumerate(getIndices())) {
    VectorType index_ty = llvm::cast<VectorType>(index.getType());
    if (index_ty.getShape() != value_ty.getShape()) {
      return emitOpError("Expected ")
             << value_ty.getShape() << " elements in indices. Got "
             << index_ty.getShape() << " in index #" << i << ".";
    }
  }
  return verifyLoadOp(*this);
}

LogicalResult VectorStoreIdxOp::verify() {
  VectorType value_ty = getValueToStore().getType();
  MemRefType ref_ty = getBase().getType();
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError(
               "Expected one index vector for each dimension of the base "
               "memref with dimension: ")
           << ref_ty.getRank() << ". Got: " << llvm::size(getIndices()) << ".";
  }
  if (value_ty.getRank() != 1) {
    return emitOpError("Expected value to have rank 1. Got: ")
           << value_ty.getRank() << ".";
  }
  for (const auto [i, index] : llvm::enumerate(getIndices())) {
    VectorType index_ty = llvm::cast<VectorType>(index.getType());
    if (index_ty.getShape() != value_ty.getShape()) {
      return emitOpError("Expected ")
             << value_ty.getShape() << " elements in indices. Got "
             << index_ty.getShape() << " in index #" << i << ".";
    }
  }
  return verifyStoreOp(*this);
}

void ReinterpretCastOp::build(OpBuilder& builder, OperationState& state,
                              Type result_type, Value input,
                              Value dynamic_offset, ValueRange dynamic_sizes) {
  state.addOperands(input);
  if (dynamic_offset) {
    state.addOperands(dynamic_offset);
  }
  state.addOperands(dynamic_sizes);
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr(
                         {1, dynamic_offset ? 1 : 0,
                          static_cast<int32_t>(dynamic_sizes.size())}));
  state.addTypes(result_type);
}

LogicalResult ReinterpretCastOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  if (source_type.getMemorySpace() != target_type.getMemorySpace()) {
    return emitOpError("Source and target memory spaces must match, but got ")
           << source_type.getMemorySpace() << " and "
           << target_type.getMemorySpace();
  }
  int64_t num_dynamic_dims = target_type.getNumDynamicDims();
  if (static_cast<int64_t>(getDynamicSizes().size()) != num_dynamic_dims) {
    return emitOpError("expected ")
           << num_dynamic_dims
           << " dynamic size(s) for the result type, but got "
           << getDynamicSizes().size();
  }
  return success();
}

OpFoldResult EraseLayoutOp::fold(FoldAdaptor adaptor) {
  // If the operand has no interesting layout then there's no need to erase it.
  if (getOperand().getType() == getType()) {
    return adaptor.getOperand();
  }
  if (auto erase_layout_op = getOperand().getDefiningOp<EraseLayoutOp>()) {
    getOperandMutable().assign(erase_layout_op.getOperand());
    return getResult();
  }
  return OpFoldResult();
}

LogicalResult EraseLayoutOp::verify() {
  MemRefType operand_type = getOperand().getType();
  MemRefType result_type = getType();
  // TODO(tlongeri): Enforce no shape changes
  if (operand_type.getElementType() != result_type.getElementType()) {
    return emitOpError("Cannot change the memref element type");
  }
  if (operand_type.getMemorySpace() != result_type.getMemorySpace() &&
      result_type.getMemorySpace()) {
    return emitOpError(
        "Memref memory space must be either erased (changed to null) or "
        "preserved");
  }
  if (operand_type.getLayout() == nullptr) {
    return emitOpError("Memref layout must be erased");
  }
  return success();
}

LogicalResult DynamicRotateOp::verify() {
  auto vty = getResult().getType();
  if (vty.getRank() <= getDimension() || getDimension() < 0) {
    return emitOpError("Invalid dimension: ") << getDimension();
  }
  if (getStride().has_value() && getStride().value() < 0) {
    return emitOpError("Rotate stride must be >= 0 if it is specified");
  }
  if (getStrideDimension().has_value() &&
      (vty.getRank() <= getStrideDimension().value() ||
       getStrideDimension().value() < 0)) {
    return emitOpError("Invalid stride dimension: ")
           << getStrideDimension().value();
  }
  if (getStride().has_value() != getStrideDimension().has_value()) {
    return emitOpError(
        "Expected either none or both stride and stride dimension are "
        "present");
  }
  return success();
}

LogicalResult ScanCountOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ScanCountOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInMask().getType());
  inferredReturnTypes.push_back(VectorType::get(
      cast<VectorType>(adaptor.getValues().getType()).getShape(),
      IntegerType::get(context, 32)));
  return success();
}

LogicalResult IotaOp::verify() {
  const int64_t rank = getType().getRank();
  SmallVector<bool> seen(rank, false);
  for (const int32_t dim : getDimensions()) {
    if (dim < 0 || dim >= getType().getRank()) {
      return emitOpError("Invalid dimension: ") << dim;
    }
    if (seen[dim]) {
      return emitOpError("Dimensions must be unique");
    }
    seen[dim] = true;
  }
  return success();
}

// a + matmul(l, r, 0) == matmul(l, r, a)
template <typename AddOp>
class CanonicalizeAddOfMatmul : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter& rewriter) const {
    auto try_canonicalize = [&](Value maybe_matmul, Value maybe_acc) {
      auto matmul = dyn_cast_if_present<MatmulOp>(maybe_matmul.getDefiningOp());
      if (!matmul || !matmul->hasOneUse()) {
        return failure();
      }
      if (auto const_acc = matmul.getAcc().getDefiningOp<arith::ConstantOp>();
          const_acc &&
          const_acc.getValue() == rewriter.getZeroAttr(const_acc.getType())) {
        IRMapping remap;
        remap.map(matmul.getAcc(), maybe_acc);
        Operation* new_matmul = rewriter.clone(*matmul, remap);
        rewriter.replaceOp(op, new_matmul->getResult(0));
        return success();
      }
      return failure();
    };
    // We tried try_canonicalize(op.getRhs(), op.getLhs()) and it caused
    // worrying numerical differences in some of kernels.
    return try_canonicalize(op.getLhs(), op.getRhs());
  }
};

LogicalResult MatmulOp::verify() {
  // Note - this is not yet an exhaustive verification of matmul. Many of the
  // invariants are spread across infer, apply, llo and below. This is,
  // however, a good start and the recommended place to add more invariants.
  const VectorType lhs_ty = getLhs().getType();
  const VectorType rhs_ty = getRhs().getType();
  const VectorType acc_ty = getAcc().getType();
  const VectorType res_ty = getResult().getType();
  if (acc_ty != res_ty) {
    return emitOpError(
        "Not implemented: matmul acc and result have different types");
  }
  if (getElementTypeBitwidth(acc_ty) != 32) {
    return emitOpError("Expected matmul acc to be 32-bit");
  }

  if (getTransposeLhs()) {
    emitOpError(
        "Lhs transpose not supported via this API - please use the "
        "dimension numbers API.");
    return failure();
  }

  if (getDimensionNumbers().has_value()) {
    auto dimension_numbers = getDimensionNumbers().value();
    auto lhs_contracting_dims = dimension_numbers.getLhsContractingDims();
    auto rhs_contracting_dims = dimension_numbers.getRhsContractingDims();
    if (lhs_contracting_dims.size() != 1) {
      emitOpError("Not implemented: lhs contracting dims must be of size 1");
      return failure();
    }
    if (rhs_contracting_dims.size() != 1) {
      emitOpError("Not implemented: rhs contracting dims must be of size 1");
      return failure();
    }

    auto lhs_contracting_dim = lhs_contracting_dims[0];
    auto rhs_contracting_dim = rhs_contracting_dims[0];

    auto lhs_batch_dims = dimension_numbers.getLhsBatchDims();
    auto rhs_batch_dims = dimension_numbers.getRhsBatchDims();

    auto lhs_non_contracting_dims =
        dimension_numbers.getLhsNonContractingDims();
    auto rhs_non_contracting_dims =
        dimension_numbers.getRhsNonContractingDims();

    if (!llvm::is_sorted(lhs_non_contracting_dims)) {
      emitOpError("Not implemented: lhs non contracting dims must be sorted");
      return failure();
    }
    if (!llvm::is_sorted(rhs_non_contracting_dims)) {
      emitOpError("Not implemented: rhs non contracting dims must be sorted");
      return failure();
    }

    if (lhs_contracting_dims.size() + lhs_non_contracting_dims.size() +
            lhs_batch_dims.size() !=
        lhs_ty.getShape().size()) {
      emitOpError(
          "Not implemented: lhs contracting + non contracting + batch dims "
          "must be of the same size as the lhs shape");
      return failure();
    }
    if (rhs_contracting_dims.size() + rhs_non_contracting_dims.size() +
            rhs_batch_dims.size() !=
        rhs_ty.getShape().size()) {
      emitOpError(
          "Not implemented: rhs contracting + non contracting + batch dims "
          "must be of the same size as the rhs shape");
      return failure();
    }

    if (lhs_ty.getShape()[lhs_contracting_dim] !=
        rhs_ty.getShape()[rhs_contracting_dim]) {
      emitOpError(
          "Not implemented: lhs and rhs contracting dims must be of the same "
          "size");
      return failure();
    }

    if (lhs_batch_dims.size() != rhs_batch_dims.size()) {
      emitOpError(
          "Not implemented: lhs and rhs should have the same number of batch "
          "dims");
      return failure();
    }
    if (lhs_batch_dims.size() > 1) {
      emitOpError("Not implemented: Up to 1 batch dim supported");
      return failure();
    }

    int64_t lhs_rank = lhs_ty.getShape().size();
    int64_t rhs_rank = rhs_ty.getShape().size();

    std::vector<bool> seen_dims_lhs(lhs_rank, false);
    std::vector<bool> seen_dims_rhs(rhs_rank, false);

    auto check_and_mark_dims = [&](const std::vector<int64_t>& dims,
                                   std::vector<bool>& seen_dims,
                                   const std::string_view operand) {
      for (int64_t dim : dims) {
        if (dim < 0 || dim >= seen_dims.size()) {
          emitOpError("Illegal: Dim ")
              << dim << " is out of bounds for " << operand;
          return failure();
        }
        if (seen_dims[dim]) {
          emitOpError("Illegal: Dim ")
              << dim << " repeats in dimension numbers of " << operand;
          return failure();
        }
        seen_dims[dim] = true;
      }
      return success();
    };

    if (failed(
            check_and_mark_dims(lhs_contracting_dims, seen_dims_lhs, "lhs")) ||
        failed(check_and_mark_dims(lhs_non_contracting_dims, seen_dims_lhs,
                                   "lhs")) ||
        failed(check_and_mark_dims(lhs_batch_dims, seen_dims_lhs, "lhs"))) {
      return failure();
    }

    if (failed(
            check_and_mark_dims(rhs_contracting_dims, seen_dims_rhs, "rhs")) ||
        failed(check_and_mark_dims(rhs_non_contracting_dims, seen_dims_rhs,
                                   "rhs")) ||
        failed(check_and_mark_dims(rhs_batch_dims, seen_dims_rhs, "rhs"))) {
      return failure();
    }

    for (int64_t dim = 0; dim < lhs_rank; ++dim) {
      if (!seen_dims_lhs[dim]) {
        emitOpError("Illegal: Dim ")
            << dim << " is not seen in lhs dimension numbers";
        return failure();
      }
    }
    for (int64_t dim = 0; dim < rhs_rank; ++dim) {
      if (!seen_dims_rhs[dim]) {
        emitOpError("Illegal: Dim ")
            << dim << " is not seen in rhs dimension numbers";
      }
    }

    const std::optional<int64_t> batch_dim_lhs =
        lhs_batch_dims.empty() ? std::nullopt
                               : std::optional<int64_t>(lhs_batch_dims[0]);
    const std::optional<int64_t> batch_dim_rhs =
        rhs_batch_dims.empty() ? std::nullopt
                               : std::optional<int64_t>(rhs_batch_dims[0]);

    // Invariant above enforces only 1 batch dim atm.
    std::optional<int64_t> batch_size = std::nullopt;
    if (batch_dim_lhs.has_value()) {
      batch_size = lhs_ty.getShape()[batch_dim_lhs.value()];
      auto rhs_batch_size = rhs_ty.getShape()[batch_dim_rhs.value()];
      if (batch_size != rhs_batch_size) {
        emitOpError("Not Implemented: batch dims must be equal");
        return failure();
      }
      if (batch_size == 0) {
        emitOpError("Illegal: batch size must be > 0");
        return failure();
      }
    }
    auto output_dim_order = dimension_numbers.getOutputDimOrder();
    if (output_dim_order.size() % 2 != 0) {
      emitOpError(
          "Illegal: output dim order must have an even number of elements.");
      return failure();
    }

    // Invariants above enforce a single batch idx for now. Future extension to
    // this will be to support multiple batch dims.

    // Verify that the output dim order is always in the form of [0,
    // lhs_batch_dims, 0, lhs_non_contracting_dims, 1,
    // rhs_non_contracting_dims].
    llvm::SmallVector<int64_t> expected_output_dim_order;
    expected_output_dim_order.reserve(2 * (lhs_batch_dims.size() +
                                           lhs_non_contracting_dims.size() +
                                           rhs_non_contracting_dims.size()));
    for (int64_t dim : lhs_batch_dims) {
      expected_output_dim_order.push_back(0);
      expected_output_dim_order.push_back(dim);
    }
    for (int64_t dim : lhs_non_contracting_dims) {
      expected_output_dim_order.push_back(0);
      expected_output_dim_order.push_back(dim);
    }
    for (int64_t dim : rhs_non_contracting_dims) {
      expected_output_dim_order.push_back(1);
      expected_output_dim_order.push_back(dim);
    }
    if (!absl::c_equal(output_dim_order, expected_output_dim_order)) {
      emitOpError(
          "Illegal: output dim order must be in the form of [0, "
          "lhs_batch_dims, 0, lhs_non_contracting_dims, 1, "
          "rhs_non_contracting_dims]");
      return failure();
    }
  }
  return success();
}

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<CanonicalizeAddOfMatmul<arith::AddFOp>,
              CanonicalizeAddOfMatmul<arith::AddIOp>>(context);
}

LogicalResult MaskCastOp::verify() {
  auto input_ty = getInput().getType();
  auto output_ty = getResult().getType();
  return success(input_ty.getShape().take_front(2) ==
                 output_ty.getShape().take_front(2));
}

LogicalResult ScanOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (issuing_core != CoreType::kScVectorSubcore) {
    return emitOpError("Scan is supported only on the SC vector subcore");
  }

  VectorType input_ty = getInput().getType();
  VectorType output_ty = getOutput().getType();

  if (input_ty.getElementType().isInteger(1)) {
    if (!output_ty.getElementType().isInteger(32)) {
      return emitOpError(
          "Output element type must be i32 vector for i1 vector inputs.");
    }
  } else {
    if (input_ty.getElementType() != output_ty.getElementType()) {
      return emitOpError("Input and output element type mismatch.");
    }
  }

  if (input_ty.getShape() != output_ty.getShape()) {
    return emitOpError("Input and output shape mismatch. Input shape: (")
           << input_ty.getShape() << "). Output shape: ("
           << output_ty.getShape() << ").";
  }

  if (input_ty.getRank() > 2) {
    return emitOpError("Input must be a rank 1 or 2 vector.");
  }

  if (input_ty.getElementType().isInteger(1) &&
      getKind() != ReductionKind::kSum) {
    return emitOpError("Only sum reduction is supported for i1 vector inputs.");
  } else if (getKind() != ReductionKind::kSum &&
             getKind() != ReductionKind::kMax &&
             getKind() != ReductionKind::kMin) {
    return emitOpError("Only sum, max and min reductions are supported.");
  }

  if (getMask() == nullptr) {
    return success();
  } else if (input_ty.getElementType().isInteger(1)) {
    return emitOpError("Mask is not supported for i1 vector inputs.");
  }

  VectorType mask_ty = getMask().getType();
  if (mask_ty.getRank() != 1) {
    return emitOpError("Mask must be a rank 1 vector.");
  }
  if (mask_ty.getShape()[0] != input_ty.getShape()[input_ty.getRank() - 1]) {
    return emitOpError("Mask and input mismatch. Expected mask of length: ")
           << input_ty.getShape()[input_ty.getRank() - 1] << ", but got "
           << mask_ty.getShape()[0] << ".";
  }

  return success();
}

LogicalResult SortOp::verify() {
  VectorType keys_ty = getKeys().getType();
  VectorType values_ty = getValues().getType();
  if (keys_ty.getShape() != values_ty.getShape()) {
    return emitOpError("Key and value shapes must match: ")
           << keys_ty.getShape() << " vs " << values_ty.getShape();
  }
  if (getMask()) {
    VectorType mask_ty = getMask().getType();
    if (keys_ty.getShape() != mask_ty.getShape()) {
      return emitOpError("Key and input mask shapes must match: ")
             << keys_ty.getShape() << " vs " << mask_ty.getShape();
    }
  }
  VectorType output_mask_ty = getOutputMask().getType();
  if (keys_ty.getShape() != output_mask_ty.getShape()) {
    return emitOpError("Key and output mask shapes must match: ")
           << keys_ty.getShape() << " vs " << output_mask_ty.getShape();
  }
  if (keys_ty != getSortedKeys().getType()) {
    return emitOpError("Key and sorted_key types must match: ")
           << keys_ty << " vs " << getSortedKeys().getType();
  }
  if (values_ty != getSortedValues().getType()) {
    return emitOpError("Value and sorted_value types must match: ")
           << values_ty << " vs " << getSortedValues().getType();
  }
  return success();
}

LogicalResult GetBarrierSemaphoreOp::verify() {
  MemRefType sem_type = getResult().getType();
  if (sem_type.getRank() != 0) {
    emitOpError("Barrier semaphore reference must be rank 0");
    return failure();
  }
  return success();
}

void SemaphoreSignalOp::build(OpBuilder& builder, OperationState& state,
                              Value semaphore, Value amount, Value device_id,
                              Value core_id) {
  build(builder, state, semaphore, amount, device_id, core_id,
        /*subcore_id=*/nullptr);
}

mlir::tpu::CoreType SemaphoreSignalOp::getTargetCoreType() {
  return getRefCoreType(getSemaphore()).value_or(GetCoreTypeOfParentOp(**this));
}

LogicalResult SemaphoreSignalOp::verify() {
  MemRefType sem_type = getSemaphore().getType();
  if (sem_type.getRank() != 0) {
    return emitOpError("Semaphore reference must be rank 0");
  }

  CoreType issuing_core_type = GetCoreTypeOfParentOp(**this);
  CoreType target_core_type = getTargetCoreType();

  if (getCoreId() == nullptr && getDeviceId() == nullptr) {
    if (target_core_type != issuing_core_type) {
      return emitOpError(
          absl::StrFormat("Target core type (%s) must match source core type "
                          "(%s) when device_id and core_id are not specified",
                          stringifyCoreType(target_core_type),
                          stringifyCoreType(issuing_core_type)));
    }
  }
  // Subcore ID applies only to SC vector subcore semaphore ops.
  if (target_core_type != CoreType::kScVectorSubcore &&
      getSubcoreId() != nullptr) {
    return emitOpError(
        "Subcore id should not be set unless target core type is SC vector "
        "subcore");
  }
  return success();
}

LogicalResult SemaphoreWaitOp::verify() {
  MemRefType sem_type = getSemaphore().getType();
  if (sem_type.getRank() != 0) {
    return emitOpError("Semaphore reference must be rank 0");
  }
  return success();
}

void EnqueueDMAOp::build(OpBuilder& builder, OperationState& state,
                         Value source, Value source_semaphore, Value target,
                         Value target_semaphore, Value device_id, Value core_id,
                         uint32_t priority, bool strict_ordering) {
  build(builder, state, source, source_semaphore, target, target_semaphore,
        device_id, core_id, /*subcore_id=*/nullptr, priority, strict_ordering);
}
mlir::tpu::CoreType EnqueueDMAOp::getTargetCoreType() {
  return getRefCoreType(getTargetSemaphore())
      .value_or(GetCoreTypeOfParentOp(**this));
}

LogicalResult EnqueueDMAOp::verify() {
  Value target_sem = getTargetSemaphore();
  if (!target_sem) {
    // TODO: b/501204503 - Support optional source and destination semaphores.
    return emitOpError("EnqueueDMA target semaphore must be provided.");
  }
  MemRefType target_sem_type = getMemRefType(target_sem);
  if (target_sem_type.getRank() != 0) {
    return emitOpError("DMA target semaphore must be rank 0");
  }
  Type target_sem_elem_type = target_sem_type.getElementType();

  Value source_sem = getSourceSemaphore();
  if (source_sem) {
    MemRefType source_sem_type = getMemRefType(source_sem);
    if (source_sem_type.getRank() != 0) {
      return emitOpError("DMA source semaphore reference must be rank 0");
    }
    if (source_sem_type.getElementType() != target_sem_elem_type) {
      return emitOpError(
          "DMA source and target semaphore must have the same type");
    }
  }
  MemRefType source_ty = getMemRefType(getSource());
  MemRefType target_ty = getMemRefType(getTarget());
  if (source_ty.getElementType() != target_ty.getElementType()) {
    return emitOpError("DMA source and target element type mismatch");
  }
  if (source_ty.getShape() != target_ty.getShape()) {
    return emitOpError("DMA source and target shape mismatch.");
  }

  if (getDeviceId() || getCoreId()) {
    if (!getSourceSemaphore()) {
      return emitOpError(
          "DMA source semaphore must be specified when device_id or core_id is "
          "specified");
    }
  }
  bool is_remote = getDeviceId() || getCoreId();
  if (getSourceSemaphore()) {
    // TODO: b/501204503 - Support optional source and destination semaphores.
    if (!is_remote) {
      return emitOpError(
          "DMA destination device_id or core_id must be specified when source "
          "semaphore is specified");
    }
  }
  int priority = getPriority();
  if (priority < 0 || priority > 1) {
    return emitOpError(
               "Not implemented: only support priority 0 or 1, but got ")
           << priority;
  }
  if (priority != 0 && is_remote) {
    return emitOpError(
        "Not implemented: non-zero priority is not supported for remote DMA");
  }
  // If the target core_type is different from the issuing core_type,
  // the specific core_id must be provided. The device_id is irrelevant here.
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  CoreType target_core = getTargetCoreType();
  if (getSourceSemaphore() &&
      getRefCoreType(getSourceSemaphore()).value_or(issuing_core) !=
          issuing_core) {
    return emitOpError("Source semaphore and source ref core type mismatched");
  }
  if (target_core != issuing_core && getCoreId() == nullptr) {
    return emitOpError(
        absl::StrFormat("Core id must be specified when target core type (%v) "
                        "is different from source core type (%v)",
                        target_core, issuing_core));
  }
  if (getStrictOrdering() && issuing_core != CoreType::kScScalarSubcore &&
      issuing_core != CoreType::kScVectorSubcore) {
    return emitOpError(
        "Strict ordering is only supported on the SC scalar and vector "
        "subcores");
  }
  if (isa<SemaphoreType>(target_sem_elem_type)) {
    if (HasMemorySpace(source_ty, MemorySpace::kSmem) ||
        HasMemorySpace(target_ty, MemorySpace::kSmem)) {
      return emitOpError(
          "Non-DMA semaphores are not supported for DMAs involving SMEM");
    }
  }
  // Subcore ID applies only to SC vector subcore DMAs.
  if (target_core != CoreType::kScVectorSubcore && getSubcoreId() != nullptr) {
    return emitOpError(
        "Subcore id should not be set unless target core type is SC vector "
        "subcore");
  }
  return success();
}

FailureOr<bool> EnqueueIndirectDMAOp::isGather() {
  const auto source_ms = dyn_cast_if_present<MemorySpaceAttr>(
      getMemRefType(getSource()).getMemorySpace());
  const auto target_ms = dyn_cast_if_present<MemorySpaceAttr>(
      getMemRefType(getTarget()).getMemorySpace());
  if (source_ms == nullptr || target_ms == nullptr) {
    return emitOpError("Source and target memory spaces must be provided");
  }
  return mlir::tpu::isGather(*getOperation(), source_ms.getValue(),
                             source_ms.getCoreType(), target_ms.getValue(),
                             target_ms.getCoreType());
}

LogicalResult EnqueueIndirectDMAOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (issuing_core != CoreType::kScVectorSubcore) {
    return emitOpError(
        "Enqueue indirect DMA is supported only on the SC vector subcore");
  }

  const MemRefType source_ty = getMemRefType(getSource());
  const MemRefType target_ty = getMemRefType(getTarget());

  if (source_ty.getElementType() != target_ty.getElementType()) {
    return emitOpError("Source and target element type mismatch");
  }

  FAILUREOR_ASSIGN_OR_RETURN(bool is_gather, isGather());

  const Value offsets = getOffsets();
  ArrayRef<int64_t> offsets_shape;
  if (auto offsets_ty = dyn_cast<MemRefType>(offsets.getType());
      offsets_ty != nullptr) {
    if (!HasMemorySpace(offsets_ty, MemorySpace::kVmem)) {
      return emitOpError("Offsets memref must be in VMEM");
    }
    offsets_shape = offsets_ty.getShape();
  } else if (auto offsets_ty = dyn_cast<VectorType>(offsets.getType());
             offsets_ty != nullptr) {
    offsets_shape = offsets_ty.getShape();
  } else {
    return emitOpError("Offsets must be a memref or vector type");
  }

  if (MemRefType sem_ty = getMemRefType(getSemaphore());
      sem_ty.getRank() != 0) {
    return emitOpError("Semaphore must be rank 0");
  }

  if (is_gather) {
    return verifyGather(getOperation(), /*operand_shape=*/source_ty.getShape(),
                        /*offsets_shape=*/offsets_shape,
                        /*results_memory_space=*/target_ty.getShape());
  }
  return verifyScatter(getOperation(), /*updates_ty=*/source_ty.getShape(),
                       /*offsets_shape=*/offsets_shape,
                       /*operand_ty=*/target_ty.getShape());
}

void WaitDMA2Op::build(OpBuilder& builder, OperationState& state,
                       Value semaphore, Value src, Value dst) {
  build(builder, state, semaphore, src, dst, /*device_id=*/nullptr,
        /*core_id=*/nullptr);
}

LogicalResult WaitDMA2Op::verify() {
  MemRefType sem_type = getSemaphore().getType();
  if (sem_type.getRank() != 0) {
    return emitOpError("DMA wait semaphore must be rank 0");
  }

  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (getRefCoreType(getSemaphore()).value_or(issuing_core) != issuing_core) {
    return emitOpError("Can only await semaphores attached to the local core");
  }

  return success();
}

LogicalResult WaitDMAOp::verify() {
  bool wait_destination = getWaitTarget();
  TypedValue<MemRefType> sem =
      wait_destination ? getTargetSemaphore() : getSourceSemaphore();
  if (!sem) {
    // TODO: b/501204503 - Support optional source and destination semaphores
    // with global reserved semaphore allocation tracking.
    return emitOpError("The awaited semaphore must be provided");
  }

  if (getMemRefType(sem).getRank() != 0) {
    return emitOpError("DMA wait semaphore must be rank 0");
  }

  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (getRefCoreType(sem).value_or(issuing_core) != issuing_core) {
    return emitOpError("Can only await semaphores attached to the local core");
  }

  // Subcore ID applies only to SC vector subcore.
  if (issuing_core != CoreType::kScVectorSubcore && getSubcoreId() != nullptr) {
    return emitOpError(
        "Subcore id should not be set unless issuing core type is SC vector "
        "subcore");
  }

  return success();
}

FailureOr<bool> WaitIndirectDMAOp::isGather() {
  const auto source_ms = dyn_cast_if_present<MemorySpaceAttr>(
      getMemRefType(getSrc()).getMemorySpace());
  const auto target_ms = dyn_cast_if_present<MemorySpaceAttr>(
      getMemRefType(getDst()).getMemorySpace());
  if (source_ms == nullptr || target_ms == nullptr) {
    return emitOpError("Source and target memory spaces must be provided");
  }
  return mlir::tpu::isGather(*getOperation(), source_ms.getValue(),
                             source_ms.getCoreType(), target_ms.getValue(),
                             target_ms.getCoreType());
}

LogicalResult WaitIndirectDMAOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  if (issuing_core != CoreType::kScVectorSubcore) {
    return emitOpError(
        "Wait indirect DMA is supported only on the SC vector subcore");
  }
  MemRefType sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    return emitOpError("Indirect DMA wait semaphore must be rank 0");
  }
  return isGather();
}

LogicalResult RegionOp::verify() {
  for (auto result_type : getResultTypes()) {
    if (!isa<FloatType, IntegerType, VectorType, IndexType>(result_type)) {
      return emitOpError(
          "Region result must be float, int, index or a vector type.");
    }
  }
  return success();
}

LogicalResult ShuffledLoadOp::verify() {
  if (getBase().getType().getRank() != getIndices().size()) {
    return emitOpError("Base memref's rank and indices size do not match: ")
           << getBase().getType().getRank() << " vs " << getIndices().size();
  }
  if (getSublaneMask().size() != getType().getShape()[0]) {
    return emitOpError("Expected sublane mask size equals to ")
           << getType().getShape()[0] << " but got " << getSublaneMask().size();
  }
  if (getSublaneOffsets().size() != getType().getShape()[0]) {
    return emitOpError("Expected sublane offsets size equals to ")
           << getType().getShape()[0] << " but got "
           << getSublaneOffsets().size();
  }
  return success();
}

LogicalResult ShuffledLoadOp::canonicalize(ShuffledLoadOp op,
                                           PatternRewriter& rewriter) {
  bool can_convert_to_simple_load = true;
  for (int i = 0; i < op.getSublaneOffsets().size(); ++i) {
    if (op.getSublaneOffsets()[i] != i) {
      can_convert_to_simple_load = false;
      break;
    };
  }
  if (can_convert_to_simple_load) {
    rewriter.replaceOpWithNewOp<tpu::LoadOp>(
        op, op.getType(), op.getBase(), op.getIndices(), op.getSublaneMask());
  }
  return success();
}

LogicalResult ShuffledStoreOp::verify() {
  if (getBase().getType().getRank() != getIndices().size()) {
    return emitOpError("Base memref's rank and indices size do not match: ")
           << getBase().getType().getRank() << " vs " << getIndices().size();
  }
  if (getValueToStore().getType().getRank() != getIndices().size()) {
    return emitOpError(
               "The rank of value to store and indices size do not match: ")
           << getBase().getType().getRank() << " vs " << getIndices().size();
  }
  if (getSublaneMask().size() != getValueToStore().getType().getShape()[0]) {
    return emitOpError("Expected sublane mask size equals to ")
           << getValueToStore().getType().getShape()[0] << " but got "
           << getSublaneMask().size();
  }
  if (getSublaneOffsets().size() != getValueToStore().getType().getShape()[0]) {
    return emitOpError("Expected sublane offsets size equals to ")
           << getValueToStore().getType().getShape()[0] << " but got "
           << getSublaneOffsets().size();
  }
  return success();
}

LogicalResult ShuffledStoreOp::canonicalize(ShuffledStoreOp op,
                                            PatternRewriter& rewriter) {
  bool can_convert_to_simple_store = true;
  for (int i = 0; i < op.getSublaneOffsets().size(); ++i) {
    if (op.getSublaneOffsets()[i] != i) {
      can_convert_to_simple_store = false;
      break;
    };
  }
  if (can_convert_to_simple_store) {
    rewriter.replaceOpWithNewOp<tpu::StoreOp>(op, op.getValueToStore(),
                                              op.getBase(), op.getIndices(),
                                              op.getSublaneMask(),
                                              /*mask=*/nullptr);
  }
  return success();
}

LogicalResult FPToSIOp::canonicalize(FPToSIOp op, PatternRewriter& rewriter) {
  if (auto round_op = op.getIn().getDefiningOp<mlir::math::RoundEvenOp>()) {
    rewriter.replaceOpWithNewOp<tpu::FPToSIOp>(
        op, op.getType(), round_op.getOperand(),
        tpu::RoundingMode::kToNearestEven);
    return success();
  }
  return failure();
}

LogicalResult ConcatenateOp::verify() {
  auto dimension = getDimension();
  if (getOperands().size() < 2) {
    return emitOpError("Expected at least 2 operands for concatenate op.");
  }
  auto first_type = cast<VectorType>(getOperand(0).getType());
  auto first_shape = first_type.getShape();
  auto first_dtype = first_type.getElementType();
  for (auto operand : getOperands()) {
    auto vty = dyn_cast<VectorType>(operand.getType());
    if (!vty) {
      return emitOpError("Operand must be a vector type.");
    }
    auto shape = vty.getShape();
    auto dtype = vty.getElementType();
    if (dtype != first_dtype) {
      return emitOpError(
          "Not implemented:: Expected all operands to have the same element "
          "type.");
    }
    for (int dim = 0; dim < shape.size(); ++dim) {
      if (dim != dimension && shape[dim] != first_shape[dim]) {
        return emitOpError(
            "Not implemented: Expected all operands to have "
            "the same shape outside of the concat dim");
      }
    }
  }
  return success();
}

/*static*/ LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::PropertyRef properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
  ConcatenateOpAdaptor adaptor(operands, attributes, properties, regions);
  auto dimension = adaptor.getDimension();
  for (auto [i, operand] : llvm::enumerate(operands)) {
    if (auto operand_ty = dyn_cast<VectorType>(operand.getType());
        !operand_ty || operand_ty.getRank() <= dimension) {
      return failure();
    }
  }
  auto first_type = cast<VectorType>(operands[0].getType());
  llvm::SmallVector<int64_t> result_shape =
      llvm::to_vector(first_type.getShape());
  Type result_dtype = first_type.getElementType();
  for (int i = 1; i < operands.size(); ++i) {
    result_shape[dimension] +=
        cast<VectorType>(operands[i].getType()).getDimSize(dimension);
  }
  inferredReturnTypes.push_back(VectorType::get(result_shape, result_dtype));
  return success();
}

LogicalResult LogOp::verify() {
  CoreType logging_core = GetCoreTypeOfParentOp(**this);
  bool is_sc_core = logging_core == CoreType::kScScalarSubcore ||
                    logging_core == CoreType::kScVectorSubcore;
  if (is_sc_core && getFormattedAttr() != nullptr &&
      getFormattedAttr().getValue()) {
    return emitOpError("Formatted logging is not supported on SC");
  }
  if (is_sc_core && getInputs().size() > 1) {
    return emitOpError("SC logging only supports 0 or 1 inputs");
  }
  if (logging_core == CoreType::kScScalarSubcore) {
    for (mlir::Value input : getInputs()) {
      if (llvm::isa<VectorType>(input.getType())) {
        return emitOpError(
            "SC scalar subcore does not support logging vectors");
      }
    }
  }
  return success();
}

LogicalResult WeirdOp::verify() {
  const mlir::Type in_type = getInput().getType();
  if (const auto in_vec_type = dyn_cast<VectorType>(in_type)) {  // Vector case.
    if (!in_vec_type.getElementType().isF32()) {
      return emitOpError("Input type must be F32");
    }
    const mlir::Type out_type = getResult().getType();
    const auto out_vec_type = dyn_cast<VectorType>(out_type);
    if (!out_vec_type) {
      return emitOpError("Output type must be a vector when input is a vector");
    }
    if (!out_vec_type.getElementType().isInteger(1)) {
      return emitOpError("Output type must be I1");
    }
  } else {  // Scalar case.
    if (!in_type.isF32()) {
      return emitOpError("Input type must be F32");
    }
    const mlir::Type out_type = getResult().getType();
    if (!out_type.isInteger(1)) {
      return emitOpError("Output type must be I1 scalar");
    }
  }
  return success();
}

LogicalResult LogBufferOp::verify() {
  const MemRefType input_type = getInput().getType();
  if (input_type.getRank() != getShape().size()) {
    return emitOpError(
        "Shape must have the same length as the rank of the input");
  }
  return success();
}

LogicalResult ReciprocalOp::verify() {
  if (!getType().getElementType().isF32()) {
    return emitOpError("Not implemented: Reciprocal op for non-f32 dtypes");
  }
  return success();
}

LogicalResult UnpackSubelementsOp::verify() {
  const int packing_factor = getElementTypeBitwidth(getType()) /
                             getElementTypeBitwidth(getSource().getType());
  if (auto index = getIndex(); index >= packing_factor) {
    return emitOpError("Index must be between 0 and the packing factor (")
           << packing_factor << "), got " << index;
  }
  if (getUnsignedIntegers() &&
      !getSource().getType().getElementType().isSignlessInteger()) {
    return emitOpError(
        "unsigned_integers can only be set when the source type is an integer");
  }
  return success();
}

LogicalResult UnpackSubelementsOp::canonicalize(UnpackSubelementsOp op,
                                                PatternRewriter& rewriter) {
  auto src_elem_ty = op.getSource().getType().getElementType();
  auto dst_elem_ty = op.getType().getElementType();
  if (!src_elem_ty.isSignlessInteger() || !dst_elem_ty.isSignlessInteger()) {
    return failure();
  }
  if (!op.getIntegerExtended()) {
    // Unpack of pack with the same format is reversible if not sign extended.
    if (auto pack = op.getSource().getDefiningOp<PackSubelementsOp>();
        pack && pack.getPackFormat() == op.getPackFormat() &&
        pack.getSources().front().getType() == op.getType()) {
      Value source = fillPositions(
          pack.getSources(), pack.getPositions(),
          getElementTypeBitwidth(op.getType()) /
              getElementTypeBitwidth(pack.getType()))[op.getIndex()];
      if (source) {
        rewriter.replaceAllOpUsesWith(op, source);
        return success();
      }
    }
    return failure();
  }
  // Set `sign_extended` to false if it's used by pack that reduces the source
  // bitwidth.
  for (auto user : op->getUsers()) {
    auto pack = dyn_cast<PackSubelementsOp>(user);
    if (!pack) {
      return failure();
    }
    auto packed_elem_ty = pack.getType().getElementType();
    if (!packed_elem_ty.isSignlessInteger() ||
        getTypeBitwidth(packed_elem_ty) > getTypeBitwidth(src_elem_ty)) {
      return failure();
    }
  }
  rewriter.modifyOpInPlace(op, [&]() { op.setIntegerExtended(false); });
  return success();
}

void PackSubelementsOp::build(OpBuilder& builder, OperationState& state,
                              const VectorType output_type,
                              const ArrayRef<Value> padded_sources,
                              const PackFormat pack_format,
                              bool unsigned_integers) {
  SmallVector<Value> sources;
  SmallVector<int32_t> positions;
  for (size_t i = 0; i < padded_sources.size(); ++i) {
    if (padded_sources[i] != nullptr) {
      sources.push_back(padded_sources[i]);
      positions.push_back(i);
    }
  }
  build(builder, state, output_type, sources, positions, pack_format,
        unsigned_integers);
}

void PackSubelementsOp::build(OpBuilder& builder, OperationState& state,
                              const VectorType output_type,
                              const ArrayRef<Value> padded_sources,
                              const PackFormat pack_format) {
  build(builder, state, output_type, padded_sources, pack_format,
        /*unsigned_integers=*/false);
}

LogicalResult PackSubelementsOp::verify() {
  return verifyPackOp(*this,
                      getElementTypeBitwidth(getSources().front().getType()) /
                          getElementTypeBitwidth(getType()));
}

void PackMaskOp::build(OpBuilder& builder, OperationState& state,
                       const VectorType output_type,
                       const ArrayRef<Value> padded_sources) {
  SmallVector<Value> sources;
  SmallVector<int32_t> positions;
  for (size_t i = 0; i < padded_sources.size(); ++i) {
    if (padded_sources[i] != nullptr) {
      sources.push_back(padded_sources[i]);
      positions.push_back(i);
    }
  }
  build(builder, state, output_type, sources, positions);
}

LogicalResult PackMaskOp::verify() {
  auto getMaskPackingFactor = [](VectorType vty) -> int64_t {
    if (vty.getRank() == 2) {
      return 1;
    }
    return vty.getDimSize(2);
  };
  return verifyPackOp(*this, getMaskPackingFactor(getType()) /
                                 getMaskPackingFactor(cast<VectorType>(
                                     getSources().front().getType())));
}

namespace {
LogicalResult verifyElementwisePacking(Operation* op, Type unpacked_ty,
                                       Type packed_ty) {
  if (!(unpacked_ty.isF32() && packed_ty.isBF16()) &&
      !(unpacked_ty.isSignlessInteger() && packed_ty.isSignlessInteger())) {
    return op->emitOpError(
        "Only packing/unpacking f32 <-> bf16 and integer <-> integer is "
        "supported");
  }
  return success();
}
}  // namespace

LogicalResult PackElementwiseOp::verify() {
  if (getSources().empty()) {
    return emitOpError("At least one source is required");
  }
  const auto src_vty = cast<VectorType>(getSources().front().getType());
  if (getElementTypeBitwidth(src_vty) != getElementTypeBitwidth(getType())) {
    return emitOpError("All sources must have the same bitwidth as the result");
  }
  if (!getType().getElementType().isSignlessInteger()) {
    return emitOpError("Output type must be a signless integer type");
  }

  auto src_elem_ty = src_vty.getElementType();
  auto tgt_elem_ty = getTargetType();
  if (failed(verifyElementwisePacking(*this, /*unpacked_ty=*/src_elem_ty,
                                      /*packed_ty=*/tgt_elem_ty))) {
    return failure();
  }
  const int packing_factor =
      getElementTypeBitwidth(src_vty) / getTypeBitwidth(tgt_elem_ty);
  if (packing_factor != getSources().size()) {
    return emitOpError("The number of sources must match the packing factor (")
           << packing_factor << "), got " << getSources().size();
  }
  return success();
}

LogicalResult UnpackElementwiseOp::verify() {
  if (failed(verifyElementwisePacking(
          *this, /*unpacked_ty=*/getType().getElementType(),
          /*packed_ty=*/getSourceType()))) {
    return failure();
  }
  const int packing_factor =
      getElementTypeBitwidth(getType()) / getTypeBitwidth(getSourceType());
  if (auto index = getIndex(); index >= packing_factor) {
    return emitOpError("Index must be between 0 and the packing factor (")
           << packing_factor << "), got " << index;
  }
  return success();
}

LogicalResult DynamicGatherOp::verify() {
  const int64_t rank = getSource().getType().getRank();
  SmallVector<bool> seen(rank, false);
  for (int32_t d : getDimensions()) {
    if (d < 0 || d >= rank) {
      return emitOpError("Dimensions must be in [0, rank), but got ") << d;
    }
    if (seen[d]) {
      return emitOpError("Dimensions must be unique");
    }
    seen[d] = true;
  }
  const ArrayRef<int64_t> source_shape = getSource().getType().getShape();
  const ArrayRef<int64_t> result_shape = getType().getShape();
  if (source_shape.size() != result_shape.size()) {
    return emitOpError("Source and result shapes must have the same rank");
  }
  for (int32_t i = 0; i < source_shape.size(); ++i) {
    if (!seen[i] && source_shape[i] != result_shape[i]) {
      return emitOpError(
          "Source and result shapes must match on non-gather dimensions");
    }
  }
  return success();
}

/*static*/ LogicalResult DynamicGatherOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::PropertyRef properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
  VectorType source_vty = cast<VectorType>(operands[0].getType());
  VectorType indices_vty = cast<VectorType>(operands[1].getType());
  inferredReturnTypes.push_back(
      VectorType::get(indices_vty.getShape(), source_vty.getElementType()));
  return success();
}

/*static*/ LogicalResult DynamicGatherOp::canonicalize(
    DynamicGatherOp op, PatternRewriter& rewriter) {
  if (llvm::all_of(op.getDimensions(), [&](const int32_t d) {
        return op.getSource().getType().getDimSize(d) == 1;
      })) {
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, op.getType(),
                                                     op.getSource());
    return success();
  }
  return failure();
}

LogicalResult AllReduceOp::verify() {
  auto in_ty = getInput().getType();
  auto in_bitwidth = getElementTypeBitwidth(in_ty);
  auto out_ty = getOutput().getType();
  auto kind = getKind();

  if (in_bitwidth == 1) {
    // For mask vectors, the single (semantically scalar) result is broadcast
    // into a vector of 32-bit ints of whatever shape the target supports (not
    // necessarily the same as the input).
    if (!out_ty.getElementType().isSignlessInteger(32)) {
      return emitOpError("Vector mask all-reduce must have i32 output");
    }
    switch (kind) {
      case ReductionKind::kSum:
      case ReductionKind::kFindFirstSet:
        break;
      default:
        return emitOpError(
            "Mask all-reduce only supports sum and find_first_set kinds");
    }
    return success();
  }

  switch (kind) {
    case ReductionKind::kSum:
    case ReductionKind::kMax:
    case ReductionKind::kMin:
      if (in_ty != out_ty) {
        return emitOpError(
            "Sum, max, and min reductions must have the same "
            "input and output type");
      }
      break;
    case ReductionKind::kArgMax:
    case ReductionKind::kArgMin:
      if (in_ty.getShape() != out_ty.getShape()) {
        return emitOpError(
            "Arg_max and arg_min "
            "must have the same input and output shape");
      }
      if (!in_ty.getElementType().isF32()) {
        return emitOpError(
            "Not Implemented: Only f32 input is supported for "
            "arg_max and arg_min");
      }
      if (!out_ty.getElementType().isSignlessInteger(in_bitwidth)) {
        return emitOpError(absl::StrFormat(
            "Arg_max and arg_min must have i%d output", in_bitwidth));
      }
      break;
    case ReductionKind::kFindFirstSet:
      return emitOpError("Only i1 input is supported for find_first_set");
      break;
  }
  return success();
}

LogicalResult ReduceIndexOp::verify() {
  auto in_ty = getInput().getType();
  auto out_ty = getOutput().getType();
  auto bitwidth = getElementTypeBitwidth(in_ty);
  auto axis = getAxis();
  auto kind = getKind();
  if (kind != ReductionKind::kArgMax && kind != ReductionKind::kArgMin) {
    return emitOpError("Reduction kind must be arg_max or arg_min");
  }
  if (!in_ty.getElementType().isF32()) {
    return emitOpError(
        "Not Implemented: Only f32 input is supported for "
        "arg_max and arg_min");
  }
  if (!out_ty.getElementType().isSignlessInteger(bitwidth)) {
    return emitOpError(
        absl::StrFormat("Arg_max and arg_min must have i%d output", bitwidth));
  }

  auto in_shape = in_ty.getShape();
  auto out_shape = out_ty.getShape();
  if (axis < 0 || axis >= in_shape.size()) {
    return emitOpError("Axis must be in [0, ")
           << in_shape.size() << "), but got " << axis;
  }

  if (in_shape.size() < 2) {
    return emitOpError("Not Implemented: Only input rank > 1 is supported.");
  }
  if (out_shape.size() != in_shape.size() - 1) {
    return emitOpError("Output rank must be one less than input rank");
  }
  int out_dim = 0;
  for (int i = 0; i < in_shape.size(); ++i) {
    if (i == axis) {
      continue;
    }
    if (in_shape[i] != out_shape[out_dim]) {
      return emitOpError(
                 "Output shape must match input shape on non-reduction "
                 "dimensions. ")
             << "Output shape (" << out_shape
             << ") does not match input shape (" << in_shape
             << ") at input dimension " << i;
    }
    out_dim++;
  }
  return success();
}

LogicalResult AssumeMultipleOp::verify() {
  if (getMultiple() < 1) {
    return emitError("Multiple must be >= 1, got ") << getMultiple();
  }
  if (auto value = mlir::getConstantIntValue(getValue());
      value.has_value() && (*value % getMultiple() != 0)) {
    return emitError("Operand is a constant ")
           << *value << " that is not a multiple of " << getMultiple();
  }
  return success();
}

LogicalResult SublaneShuffleOp::verify() {
  auto lhs = getLhs();
  auto rhs = getRhs();
  auto result = getResult();
  auto lhs_ty = dyn_cast<VectorType>(lhs.getType());
  auto rhs_ty = dyn_cast<VectorType>(rhs.getType());
  auto result_ty = dyn_cast<VectorType>(result.getType());

  if (!lhs_ty || !rhs_ty || !result_ty) {
    return emitOpError("Expected operands and result to be vector types");
  }

  if (lhs_ty.getShape() != rhs_ty.getShape() ||
      lhs_ty.getShape() != result_ty.getShape()) {
    return emitOpError("Expected lhs, rhs, and result shapes to match");
  }
  if (lhs_ty.getElementType() != rhs_ty.getElementType() ||
      lhs_ty.getElementType() != result_ty.getElementType()) {
    return emitOpError("Expected lhs, rhs, and result element types to match");
  }

  auto pattern = getPattern();
  auto shape = result_ty.getShape();
  if (shape.size() < 2 || shape.size() > 3) {
    return emitOpError("Vreg rank should be 2 or 3");
  }
  auto sublane_count = shape[0];

  if (pattern.size() != sublane_count) {
    return emitOpError("Expected pattern size (")
           << pattern.size() << ") to match result/operand sublanes ("
           << sublane_count << ")";
  }

  int64_t total_input_sublanes = sublane_count * 2;
  for (int32_t idx : pattern) {
    if (idx < 0 || idx >= total_input_sublanes) {
      return emitOpError("Pattern index ") << idx << " out of bounds [0, "
                                           << (total_input_sublanes - 1) << "]";
    }
  }
  return success();
}

OpFoldResult TruncFOp::fold(FoldAdaptor adaptor) {
  auto resElemType = cast<FloatType>(getElementTypeOrSelf(getType()));
  const llvm::fltSemantics& targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr, FloatAttr::ValueType,
                         FloatAttr::ValueType, /*PoisonAttr=*/void>(
      adaptor.getOperands(), getType(),
      [this, &targetSemantics](const APFloat& a, bool& castStatus) {
        llvm::RoundingMode llvmRoundingMode =
            convertTpuRoundingModeToLLVMIR(getRoundingMode());
        FailureOr<APFloat> result =
            convertFloatValue(a, targetSemantics, llvmRoundingMode);
        if (failed(result)) {
          castStatus = false;
          return a;
        }
        return *result;
      });
}

OpFoldResult ExtFOp::fold(FoldAdaptor adaptor) {
  auto resElemType = cast<FloatType>(getElementTypeOrSelf(getType()));
  const llvm::fltSemantics& targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr, FloatAttr::ValueType,
                         FloatAttr::ValueType, /*PoisonAttr=*/void>(
      adaptor.getOperands(), getType(),
      [&targetSemantics](const APFloat& a, bool& castStatus) {
        FailureOr<APFloat> result = convertFloatValue(a, targetSemantics);
        if (failed(result)) {
          castStatus = false;
          return a;
        }
        return *result;
      });
}

LogicalResult ReshapeOp::verify() {
  auto src_ty = getSource().getType();
  auto dst_ty = getResult().getType();
  if (src_ty.getElementType() != dst_ty.getElementType()) {
    return emitOpError("element type must match");
  }
  if (src_ty.getNumElements() != dst_ty.getNumElements()) {
    return emitOpError() << "element count must match";
  }
  return success();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  // No-op reshape.
  if (getSource().getType() == getType()) {
    return getSource();
  }
  // Reshape of a reshape is a reshape.
  if (auto source_reshape = getSource().getDefiningOp<ReshapeOp>()) {
    setOperand(source_reshape.getSource());
    return getResult();
  }
  // Reshape of a constant is a constant.
  if (auto cst = dyn_cast_if_present<DenseElementsAttr>(adaptor.getSource())) {
    return cst.reshape(getType());
  }
  return nullptr;
}

LogicalResult StochasticConvertElementwiseOp::verify() {
  auto dst_ty = getDstType();
  if (!dst_ty.isBF16() &&
      !llvm::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType,
                 mlir::Float8E4M3B11FNUZType>(dst_ty)) {
    return emitOpError(
        "Only bf16, f8e5m2, f8e4m3fn, and f8e4m3b11fnuz are supported as "
        "destination types.");
  }
  return success();
}

LogicalResult FetchAndAddSyncOp::verify() {
  CoreType issuing_core = GetCoreTypeOfParentOp(**this);
  CoreType target_core = getRefCoreType(getBase()).value_or(issuing_core);
  switch (target_core) {
    case CoreType::kScVectorSubcore:
      break;
    case CoreType::kScScalarSubcore:
      return emitOpError("SC scalar subcore target not supported yet");
    default:
      return emitOpError("Only SC vector subcore is supported, got ")
             << target_core;
  }
  MemRefType base_type = getBase().getType();

  if (base_type.getRank() != getIndices().size()) {
    return emitOpError("Number of indices (")
           << getIndices().size() << ") must match memref rank ("
           << base_type.getRank() << ")";
  }
  // TODO(slebedev): Require the base to be in SMEM.
  // TODO(slebedev): Check that the enclosing function has subcore_parallel
  // in its dimension semantics.
  return success();
}

LogicalResult CoreIdOp::verify() {
  Operation* parent = GetParentOpWithCoreType(**this);
  if (parent == nullptr) {
    return emitOpError("Failed to infer the core type from parent ops");
  }
  CoreType core_type = *TPUDialect::GetCoreTypeAttr(parent);
  if (core_type == CoreType::kTc) {
    return emitOpError("Unsupported core type: ") << core_type;
  }
  // Querying Core ID is not supported under dimension semantics.
  if (auto func_op = getOperation()->getParentOfType<func::FuncOp>();
      func_op != nullptr &&
      func_op->getAttrOfType<ArrayAttr>("dimension_semantics") != nullptr) {
    return failure();
  }
  return success();
}

LogicalResult SubcoreIdOp::verify() {
  Operation* parent = GetParentOpWithCoreType(**this);
  if (parent == nullptr) {
    return emitOpError("Failed to infer the core type from parent ops");
  }
  CoreType core_type = *TPUDialect::GetCoreTypeAttr(parent);
  if (core_type == CoreType::kTc) {
    return emitOpError("Unsupported core type: ") << core_type;
  }
  // Subcore ID is not supported under dimension semantics.
  if (auto func_op = getOperation()->getParentOfType<func::FuncOp>();
      func_op != nullptr &&
      func_op->getAttrOfType<ArrayAttr>("dimension_semantics") != nullptr) {
    return failure();
  }
  return success();
}
}  // namespace tpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_ops.cc.inc"

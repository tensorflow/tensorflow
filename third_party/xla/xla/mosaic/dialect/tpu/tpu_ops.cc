/* Copyright 2023 The JAX Authors.

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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"

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
    APFloat sourceValue, const llvm::fltSemantics &targetSemantics,
    llvm::RoundingMode roundingMode = llvm::RoundingMode::NearestTiesToEven) {
  bool losesInfo = false;
  auto status = sourceValue.convert(targetSemantics, roundingMode, &losesInfo);
  if (losesInfo || status != APFloat::opOK) {
    return failure();
  }

  return sourceValue;
}

}  // namespace

LogicalResult UnrollVectorsOp::canonicalize(UnrollVectorsOp op,
                                            PatternRewriter &rewriter) {
  RollVectorsOp roll_op =
      dyn_cast_or_null<RollVectorsOp>(op.getOperand().getDefiningOp());
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
  auto in_bitwidth = in_ty.getElementTypeBitWidth();
  auto out_bitwidth = out_ty.getElementTypeBitWidth();
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

LogicalResult MemRefSliceOp::verify() {
  auto source_type = getMemRefType(getMemRef());
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
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type. Note that MemRefType does not allow a null layout so we treat
  // the default identity affine map as an "unset" value instead.
  bool is_target_memory_space_provided = target_memory_space != nullptr;
  if (is_target_memory_space_provided &&
      target_memory_space != source_type.getMemorySpace()) {
    return emitOpError(
        "Memory spaces must match if the target memory space is provided.");
  }
  if (isa<TiledLayoutAttr>(source_layout) &&
      !isa<TiledLayoutAttr>(target_layout)) {
    // TODO(slebedev): Remove this special-case once we move layout propagation
    // to the infer-memref-layout pass.
  } else if (isa<StridedLayoutAttr>(target_layout)) {
    SmallVector<int64_t> source_strides;
    int64_t source_offset;
    if (failed(
            source_type.getStridesAndOffset(source_strides, source_offset))) {
      return failure();
    }
    int64_t target_offset = source_offset;
    if (target_offset != ShapedType::kDynamic) {
      for (auto [base_idx, source_stride] :
           llvm::zip(getBaseIdx(), source_strides)) {
        if (auto idx = getConstantIntValue(base_idx)) {
          target_offset += *idx * source_stride;
        } else {
          target_offset = ShapedType::kDynamic;
          break;
        }
      }
    }
    auto expected_layout =
        StridedLayoutAttr::get(getContext(), target_offset, source_strides);
    if (target_layout != expected_layout) {
      return emitOpError("Layout mismatch: got ")
             << target_layout << ", expected " << expected_layout << ".";
    }
  } else {
    bool is_target_layout_identity_map =
        isa<AffineMapAttr>(target_layout) && target_layout.isIdentity();
    if (!is_target_layout_identity_map && target_layout != source_layout) {
      return emitOpError(
          "Layouts must match if the target layout is not an identity map.");
    }
  }
  if (getDynamicSizes().size() != target_type.getNumDynamicDims()) {
    return emitOpError(
        "Number of provided dynamic dimensions sizes must match the number of "
        "dynamic dimensions in the target type.");
  }
  return success();
}

LogicalResult MemRefSliceOp::canonicalize(MemRefSliceOp op,
                                          PatternRewriter &rewriter) {
  auto erase_layout = op.getMemRef().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout) {
    return failure();
  }
  // Push layout erasure through slicing. It is important we see the layout
  // for lowering and don't make it hard for other ops to query it.
  auto layout_ref = erase_layout.getOperand();
  MemRefType layout_ty = layout_ref.getType();
  auto new_result_type = MemRefType::get(
      op.getResult().getType().getShape(), layout_ty.getElementType(),
      layout_ty.getLayout(), layout_ty.getMemorySpace());
  auto slice =
      rewriter.create<MemRefSliceOp>(op.getLoc(), new_result_type, layout_ref,
                                     op.getBaseIdx(), op.getDynamicSizes());
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), slice);
  return success();
}

LogicalResult MemRefSqueezeOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();

  if (target_type.getMemorySpace() != nullptr &&
      target_type.getMemorySpace() != source_type.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }

  if (target_type.getElementType() != source_type.getElementType()) {
    return emitOpError("Element types don't match.");
  }

  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  auto squeezed_or =
      computeSqueezedDimsChecked(*this, source_shape, target_shape);
  if (failed(squeezed_or)) {
    return failure();
  }

  auto source_layout = source_type.getLayout();
  auto target_layout = target_type.getLayout();
  if (isa<TiledLayoutAttr>(source_layout) &&
      !isa<TiledLayoutAttr>(target_layout)) {
    // TODO(slebedev): Remove this special-case once we move layout propagation
    // to the infer-memref-layout pass.
  } else if (isa<StridedLayoutAttr>(target_layout)) {
    SmallVector<int64_t> source_strides;
    int64_t source_offset;
    if (failed(
            source_type.getStridesAndOffset(source_strides, source_offset))) {
      return failure();
    }
    SmallVector<int64_t> target_strides;
    for (auto [i, stride] : llvm::enumerate(source_strides)) {
      if (!llvm::is_contained(*squeezed_or, i)) {
        target_strides.push_back(stride);
      }
    }
    auto expected_layout =
        StridedLayoutAttr::get(getContext(), source_offset, target_strides);
    if (target_layout != expected_layout) {
      return emitOpError("Layout mismatch: got ")
             << target_layout << ", expected " << expected_layout << ".";
    }
  }

  auto erase_layout_op = getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout_op) {
    return success();
  }

  auto layout_ref = erase_layout_op.getOperand();
  MemRefType layout_ty = getMemRefType(layout_ref);
  auto layout_attr = dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  if (!layout_attr) {
    return emitOpError(
        "Input from EraseLayoutOp is expected to have a TiledLayoutAttr.");
  }
  auto &squeezed = squeezed_or.value();
  if (squeezed.empty() && source_shape != target_shape) {
    return failure();
  }

  auto tiles = layout_attr.getTiles();
  if (tiles.size() == 1) {
    auto tile = layout_attr.getTiles().front();
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
  } else {
    auto first_tile = tiles.front();
    for (int dim : squeezed) {
      int first_tiled = source_shape.size() - first_tile.dimensions().size();
      if (dim >= first_tiled) {
        return emitOpError() << "When multiple tiles are present, no tiled "
                                "dimensions can be squeezed.";
      }
    }
  }

  return success();
}

LogicalResult MemRefSqueezeOp::canonicalize(MemRefSqueezeOp op,
                                            PatternRewriter &rewriter) {
  auto source_type = getMemRefType(op.getInput());
  auto target_type = op.getType();
  auto erase_layout = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout) {
    return failure();
  }

  auto layout_ref = erase_layout.getOperand();
  MemRefType layout_ty = getMemRefType(layout_ref);
  auto layout_attr = dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  if (!layout_attr) {
    return failure();
  }

  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  auto squeezed_or = computeSqueezedDimsChecked(op, source_shape, target_shape);
  if (failed(squeezed_or)) {
    return failure();
  }
  auto &squeezed = squeezed_or.value();
  if (squeezed.empty() && source_shape != target_shape) {
    return failure();
  }

  SmallVector<int64_t> tile_strides =
      llvm::to_vector(layout_attr.getTileStrides());
  for (int i = squeezed.size() - 1; i >= 0; --i) {
    tile_strides.erase(tile_strides.begin() + squeezed[i]);
  }

  tpu::TiledLayoutAttr new_layout;
  bool target_is_1d = target_shape.size() == 1;
  auto tiles = layout_attr.getTiles();
  if (target_is_1d && tiles.size() == 1) {
    auto tile_dims = llvm::to_vector(tiles.front().dimensions());
    int first_tiled = source_shape.size() - tile_dims.size();
    for (int i = squeezed.size() - 1; i >= 0; --i) {
      int dim = squeezed[i];
      if (dim >= first_tiled) {
        int tile_idx = dim - first_tiled;
        if (tile_idx < 0 || tile_idx >= static_cast<int>(tile_dims.size())) {
          return op.emitError() << "Internal error: tile index out of bounds.";
        }
        tile_dims.erase(tile_dims.begin() + tile_idx);
      }
    }
    new_layout = tpu::TiledLayoutAttr::get(
        op.getContext(), {xla::Tile(tile_dims)}, tile_strides);
  } else {
    new_layout = tpu::TiledLayoutAttr::get(
        op.getContext(), layout_attr.getTiles(), tile_strides);
  }

  auto new_ty = MemRefType::get(target_shape, layout_ty.getElementType(),
                                new_layout, layout_ty.getMemorySpace());

  auto new_squeeze =
      rewriter.create<MemRefSqueezeOp>(op.getLoc(), new_ty, layout_ref);
  rewriter.replaceOpWithNewOp<tpu::EraseLayoutOp>(op, target_type, new_squeeze);
  return success();
}

LogicalResult RelayoutOp::verify() {
  auto in_layout_array_attr =
      getOperation()->getAttrOfType<ArrayAttr>("in_layout");
  if (!in_layout_array_attr || in_layout_array_attr.empty()) {
    return emitOpError("missing or empty 'in_layout' attribute");
  }
  if (in_layout_array_attr.size() != 1) {
    return emitOpError(
        "'in_layout' attribute must be an array containing a single "
        "VectorLayoutAttr");
  }
  auto src_vla = dyn_cast<tpu::VectorLayoutAttr>(in_layout_array_attr[0]);
  if (!src_vla) {
    return emitOpError("'in_layout' attribute is not a VectorLayoutAttr");
  }

  auto out_layout_array_attr =
      getOperation()->getAttrOfType<ArrayAttr>("out_layout");
  if (!out_layout_array_attr || out_layout_array_attr.empty()) {
    return emitOpError("missing or empty 'out_layout' attribute");
  }
  if (out_layout_array_attr.size() != 1) {
    return emitOpError(
        "'out_layout' attribute must be an array containing a single "
        "VectorLayoutAttr");
  }
  auto dst_vla = dyn_cast<tpu::VectorLayoutAttr>(out_layout_array_attr[0]);
  if (!dst_vla) {
    return emitOpError("'out_layout' attribute is not a VectorLayoutAttr");
  }

  VectorType input_type = cast<VectorType>(getInput().getType());
  VectorType output_type = cast<VectorType>(getOutput().getType());

  if (input_type.getShape() != output_type.getShape()) {
    return emitOpError("input and output shapes must match");
  }
  if (input_type.getElementType() != output_type.getElementType()) {
    // Allow i1 to i1 even if bitwidth in layout changes.
    if (!(input_type.getElementType().isInteger(1) &&
          output_type.getElementType().isInteger(1))) {
      return emitOpError(
          "input and output element types must match for non-mask relayouts");
    }
  }
  return success();
}

LogicalResult MemRefReshapeOp::verify() {
  auto src_ty = getMemRefType(getInput());
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
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  auto tgt_layout = dyn_cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (!tgt_layout) {
    return success();
  }
  auto src_layout = dyn_cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  if (!src_layout || src_layout.getTiles().empty()) {
    return emitOpError("Expected a tiled layout for the input memref.");
  }
  if (src_layout.getTiles() != tgt_layout.getTiles()) {
    return emitOpError(
        "Expected the same tiling for the input and output memref.");
  }
  auto tile = src_layout.getTiles().front().dimensions();
  if (tile.size() != 2) {
    return emitOpError("Not implemented: memref reshape with 1D tiling.");
  }
  SmallVector<int64_t> src_tile_strides(src_layout.getTileStrides());
  if (ComputeTileStrides(src_ty, tile) != src_tile_strides) {
    return emitOpError("Not implemented: reshape on a non-contiguous memref.");
  }
  auto src_tiled_shape = src_ty.getShape().take_back(2);
  auto tgt_tiled_shape = tgt_ty.getShape().take_back(2);
  bool is_src_align_tile_2nd_minor = src_tiled_shape[0] % tile[0] == 0;
  bool is_src_align_tile_minor = src_tiled_shape[1] % tile[1] == 0;
  bool is_tgt_align_tile_2nd_minor = tgt_tiled_shape[0] % tile[0] == 0;
  bool is_tgt_align_tile_minor = tgt_tiled_shape[1] % tile[1] == 0;
  if (tile[0] == 1 && is_src_align_tile_minor && is_tgt_align_tile_minor) {
    // When the tiling is (1, ?) and the source and target shapes are aligned
    // to the tile, we support reshape on any dims.
  } else if (tgt_tiled_shape[1] != src_tiled_shape[1]) {
    return emitError("Expected the minormost dimension to be unchanged");
  } else if (tgt_tiled_shape[0] != src_tiled_shape[0]) {
    if (!is_src_align_tile_2nd_minor || !is_tgt_align_tile_2nd_minor) {
      return emitError(
          "Expected the 2nd minor dimension is aligned to the tile");
    }
  }
  return success();
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

LogicalResult MemRefReshapeOp::canonicalize(MemRefReshapeOp op,
                                            PatternRewriter &rewriter) {
  auto src_ty = op.getInput().getType();
  auto dst_ty = op.getType();
  auto erase_layout_op = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout_op) {
    return failure();
  }
  auto layout_ref = erase_layout_op.getOperand();
  auto layout_ty = layout_ref.getType();
  auto layout = dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  CHECK(!layout.getTiles().empty());
  auto tile = layout.getTiles().front().dimensions();
  auto new_tile_strides = ComputeTileStrides(dst_ty, tile);
  auto new_layout = tpu::TiledLayoutAttr::get(
      src_ty.getContext(), layout.getTiles(), new_tile_strides);
  auto new_result_ty =
      MemRefType::get(dst_ty.getShape(), dst_ty.getElementType(), new_layout,
                      layout_ty.getMemorySpace());
  auto reshape =
      rewriter.create<MemRefReshapeOp>(op.getLoc(), new_result_ty, layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), reshape);
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
  auto src_bitwidth = src_ty.getElementTypeBitWidth();
  auto tgt_bitwidth = tgt_ty.getElementTypeBitWidth();
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
  // TODO(jevinjiang): verify memref tiling is valid. Here we just assume the
  // source and target tilings are valid.
  auto src_tile = src_layout.getTiles().front().dimensions();
  auto tgt_tile = tgt_layout.getTiles().front().dimensions();
  if (src_tile[0] * src_bitwidth != tgt_tile[0] * tgt_bitwidth) {
    return emitOpError("Invalid memref bitcast.");
  }
  return success();
}

LogicalResult MemRefBitcastOp::canonicalize(MemRefBitcastOp op,
                                            PatternRewriter &rewriter) {
  auto src_ty = op.getInput().getType();
  auto dst_ty = op.getType();
  if (src_ty == dst_ty) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  auto erase_layout_op = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout_op) {
    return failure();
  }
  auto src_bitwidth = src_ty.getElementTypeBitWidth();
  auto tgt_bitwidth = dst_ty.getElementTypeBitWidth();
  auto layout_ref = erase_layout_op.getOperand();
  auto layout_ty = layout_ref.getType();
  auto layout = cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  CHECK(!layout.getTiles().empty());
  auto tile = layout.getTiles().front().dimensions();
  if (tile[0] * src_bitwidth % tgt_bitwidth != 0) {
    return failure();
  }
  SmallVector<xla::Tile, 2> new_tiles = {
      xla::Tile({tile[0] * src_bitwidth / tgt_bitwidth, 128})};
  if (tgt_bitwidth < 32) {
    new_tiles.push_back(xla::Tile({32 / tgt_bitwidth, 1}));
  }
  auto new_layout = tpu::TiledLayoutAttr::get(src_ty.getContext(), new_tiles,
                                              layout.getTileStrides());
  auto new_result_ty =
      MemRefType::get(dst_ty.getShape(), dst_ty.getElementType(), new_layout,
                      layout_ty.getMemorySpace());
  auto bitcast =
      rewriter.create<MemRefBitcastOp>(op.getLoc(), new_result_ty, layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), bitcast);
  return success();
}

template <typename Op>
LogicalResult verifyStridedOp(Op op, MemRefType memref_ty,
                              VectorType vector_ty) {
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
    if (strides[i] < 1) {
      op.emitError("Strides[") << i << "]=" << strides[i] << " must be >= 1";
      return failure();
    }
  }
  return success();
}

LogicalResult StridedLoadOp::verify() {
  return verifyStridedOp<StridedLoadOp>(*this, getMemRefType(getBase()),
                                        getType());
}

LogicalResult StridedStoreOp::verify() {
  return verifyStridedOp<StridedStoreOp>(*this, getMemRefType(getBase()),
                                         getValueToStore().getType());
}

LogicalResult VectorStoreOp::verify() {
  if (!getStrides().empty()) {
    return emitError("Not implemented: general vector store with strides.");
  }
  VectorType value_ty = getValueToStore().getType();
  MemRefType ref_ty = getBase().getType();

  if (value_ty.getElementType() != ref_ty.getElementType()) {
    return emitOpError(
        "Expected base and valueToStore element type should match");
  }
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError("Expected ") << ref_ty.getRank() << " indices";
  }
  if (getMask()) {
    if (value_ty.getElementTypeBitWidth() != 32) {
      return emitError(
          "Not implemented: masked store with non-32-bit element type");
    }
    if (value_ty.getShape() != getMask().getType().getShape())
      return emitOpError("Expected valueToStore shape to match mask shape");
  }
  return success();
}

LogicalResult VectorLoadOp::verify() {
  const MemRefType ref_ty = getBase().getType();
  if (!getStrides().empty()) {
    if (llvm::size(getStrides()) != ref_ty.getRank()) {
      return emitOpError("Expected ") << ref_ty.getRank() << " strides.";
    }
    return emitError("Not implemented: general vector load with strides.");
  }
  const VectorType value_ty = getResult().getType();

  if (value_ty.getElementType() != ref_ty.getElementType()) {
    return emitOpError("Expected base and result element type to match.");
  }
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError("Expected ") << ref_ty.getRank() << " indices.";
  }
  if (getMask()) {
    if (value_ty.getElementTypeBitWidth() != 32) {
      return emitError(
          "Not implemented: masked load with non-32-bit element type");
    }
    if (vector::isBroadcastableTo(getMask().getType(), value_ty) !=
        vector::BroadcastableToResult::Success) {
      return emitOpError(
          "Expected mask shape to be broadcastable to result shape.");
    }
  }
  return success();
}

LogicalResult ReinterpretCastOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  return success(
      source_type.getMemorySpace() &&  // Require memory space annotations.
      source_type.getMemorySpace() == target_type.getMemorySpace());
}

template <typename Op>
LogicalResult verifyRotateOp(Op op) {
  auto vty = op.getResult().getType();
  if (vty.getRank() <= op.getDimension() || op.getDimension() < 0) {
    op.emitOpError("Invalid dimension: ") << op.getDimension();
    return failure();
  }
  if (op.getStride().has_value() && op.getStride().value() < 0) {
    op.emitOpError("Rotate stride must be >= 0 if it is specified");
    return failure();
  }
  if (op.getStrideDimension().has_value() &&
      (vty.getRank() <= op.getStrideDimension().value() ||
       op.getStrideDimension().value() < 0)) {
    op.emitOpError("Invalid stride dimension: ")
        << op.getStrideDimension().value();
    return failure();
  }
  if (op.getStride().has_value() != op.getStrideDimension().has_value()) {
    op.emitOpError(
        "Expected either none or both stride and stride dimension are "
        "present");
    return failure();
  }
  return success();
}

// TODO(b/347016737): deprecate static rotate
LogicalResult RotateOp::verify() { return verifyRotateOp<RotateOp>(*this); }

LogicalResult DynamicRotateOp::verify() {
  return verifyRotateOp<DynamicRotateOp>(*this);
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

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
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
        Operation *new_matmul = rewriter.clone(*matmul, remap);
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
  if (acc_ty.getElementTypeBitWidth() != 32) {
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

    auto check_and_mark_dims = [&](const std::vector<int64_t> &dims,
                                   std::vector<bool> &seen_dims,
                                   const std::string_view operand) {
      for (int64_t dim : dims) {
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
    if (batch_dim_lhs != batch_dim_rhs) {
      emitOpError("Not Implemented: batch dims must be equal");
      return failure();
    }
    if (batch_dim_lhs.has_value() && (batch_dim_lhs.value() != 0)) {
      emitOpError("Not Implemented: batch dims pos must be 0");
      return failure();
    }
    // Invariant above enforces only 1 batch dim atm, and that both are eq
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
    if (batch_size.has_value()) {
      if (output_dim_order[0] != 0 || output_dim_order[1] != 0) {
        emitOpError(
            "Not implemented: Output with batch size must be the lhs 0 idx for "
            "now.");
        return failure();
      }
    }

    // Invariants above enforce a single batch idx for now, and that it is in
    // position 0. Future extensions to this will be to:
    // 1. Support multiple batch dims
    // 2. Support batch dims in any position in the output dim order
    if (lhs_non_contracting_dims.size() != 1) {
      emitOpError(
          "Not implemented: lhs non contracting dims must be of size 1");
      return failure();
    }
    if (rhs_non_contracting_dims.size() != 1) {
      emitOpError(
          "Not implemented: rhs non contracting dims must be of size 1");
      return failure();
    }

    // A bit long winded, but the invariants we enforce below are:
    // 1. The output order idx is 0 (lhs) or 1 (rhs)
    // 2. The output dim order is in valid bounds
    // 3. We saw the rhs and lhs non contracting dims in the output dim order
    // 4. We never see the contracting dims in the output dim order
    // 5. We only see each of the non contracting dim once
    std::vector<bool> lhs_dims_seen_in_output(lhs_rank, false);
    std::vector<bool> rhs_dims_seen_in_output(rhs_rank, false);

    // Iterate over the output dimension order
    for (int dim_pos = 0; dim_pos < output_dim_order.size(); dim_pos += 2) {
      auto idx = output_dim_order[dim_pos];
      auto dim = output_dim_order[dim_pos + 1];

      if (idx != 0 && idx != 1) {
        emitOpError("Illegal: output dim order index must be 0 or 1");
        return failure();
      }
      auto is_lhs = (idx == 0);

      if (is_lhs) {
        if (dim < 0 || dim >= lhs_rank) {
          emitOpError("Illegal: lhs dimension index out of bounds");
          return failure();
        }
        if (lhs_dims_seen_in_output[dim]) {
          emitOpError("Illegal: lhs dimension ")
              << dim << " appears more than once in output dim order";
          return failure();
        }
        if (dim == lhs_contracting_dim) {
          emitOpError("Illegal: contracting dimension ")
              << dim << " appears in lhs output dim order";
          return failure();
        }
        // batch_dim_lhs is either 0 or nullopt
        if (dim == batch_dim_lhs) {
          // Upstream invariants enforce that batch dim is in position 0
          // of the output dim order.
          rhs_dims_seen_in_output[dim] = true;
        }
        lhs_dims_seen_in_output[dim] = true;
      } else {
        if (dim < 0 || dim >= rhs_rank) {
          emitOpError("Illegal: rhs dimension index out of bounds");
          return failure();
        }
        if (rhs_dims_seen_in_output[dim]) {
          emitOpError("Illegal: rhs dimension ")
              << dim << " appears more than once in output dim order";
          return failure();
        }
        if (dim == rhs_contracting_dim) {
          emitOpError("Illegal: contracting dimension ")
              << dim << " appears in rhs output dim order";
          return failure();
        }
        if (dim == batch_dim_rhs) {
          // Upstream invariants enforce that batch dim is in position 0
          // of the output dim order.
          lhs_dims_seen_in_output[dim] = true;
        }
        rhs_dims_seen_in_output[dim] = true;
      }
    }

    // Check that all dims have been seen (except contracting dims)
    for (int i = 0; i < lhs_rank; ++i) {
      if (i == lhs_contracting_dim) {
        continue;
      }
      if (!lhs_dims_seen_in_output[i]) {
        emitOpError("Illegal: lhs non-contracting dimension ")
            << i << " is not seen in output dim order";
        return failure();
      }
    }

    for (int i = 0; i < rhs_rank; ++i) {
      if (i == rhs_contracting_dim) {
        continue;
      }
      if (!rhs_dims_seen_in_output[i]) {
        emitOpError("Illegal: rhs non-contracting dimension ")
            << i << " is not seen in output dim order";
        return failure();
      }
    }
  }
  return success();
}

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<CanonicalizeAddOfMatmul<arith::AddFOp>,
               CanonicalizeAddOfMatmul<arith::AddIOp>>(context);
}

LogicalResult MaskCastOp::verify() {
  auto input_ty = getInput().getType();
  auto output_ty = getResult().getType();
  return success(input_ty.getElementType() == output_ty.getElementType() &&
                 output_ty.getRank() == 3 &&
                 (input_ty.getRank() == 2 ||
                  (input_ty.getRank() == 3 &&
                   input_ty.getDimSize(2) < output_ty.getDimSize(2))) &&
                 input_ty.getShape().take_front(2) ==
                     output_ty.getShape().take_front(2));
  return success();
}

LogicalResult GetBarrierSemaphoreOp::verify() {
  auto sem_type = getMemRefType(getResult());
  if (sem_type.getRank() != 0) {
    emitOpError("Barrier semaphore reference must be rank 0");
    return failure();
  }
  return success();
}

void SemaphoreSignalOp::build(OpBuilder &builder, OperationState &state,
                              Value semaphore, Value amount, Value device_id,
                              Value core_id) {
  build(builder, state, semaphore, amount, device_id, core_id,
        /*core_type=*/nullptr);
}

LogicalResult SemaphoreSignalOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    return emitOpError("Semaphore reference must be rank 0");
  }

  FailureOr<std::optional<CoreType>> issuing_core_type_maybe =
      GetCoreTypeOfParentFunc(**this);
  if (failed(issuing_core_type_maybe)) {
    return issuing_core_type_maybe;
  }
  CoreType issuing_core_type = issuing_core_type_maybe->value_or(CoreType::kTc);
  CoreType target_core_type = getCoreType().value_or(issuing_core_type);

  if (getCoreId() == nullptr && getDeviceId() == nullptr) {
    if (target_core_type != issuing_core_type) {
      return emitOpError(
          absl::StrFormat("Target core type (%s) must match source core type "
                          "(%s) when device_id and core_id are not specified",
                          stringifyCoreType(target_core_type),
                          stringifyCoreType(issuing_core_type)));
    }
  }
  if ((issuing_core_type == CoreType::kTc &&
       target_core_type == CoreType::kScScalarSubcore) ||
      (issuing_core_type == CoreType::kScScalarSubcore &&
       target_core_type == CoreType::kTc)) {
    return emitOpError("Signalling between TC and SC is not implemented");
  }
  return success();
}

LogicalResult SemaphoreWaitOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    return emitOpError("Semaphore reference must be rank 0");
  }
  return success();
}

LogicalResult EnqueueDMAOp::verify() {
  auto source_sem = getSourceSemaphore();
  if (source_sem) {
    auto source_sem_type = getMemRefType(getSourceSemaphore());
    if (source_sem_type.getRank() != 0) {
      return emitOpError("DMA source semaphore reference must be rank 0");
    }
  }
  auto target_sem_type = getMemRefType(getTargetSemaphore());
  if (target_sem_type.getRank() != 0) {
    return emitOpError("DMA target semaphore must be rank 0");
  }
  auto source_ty = getMemRefType(getSource());
  auto target_ty = getMemRefType(getTarget());
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
  return success();
}

LogicalResult EnqueueIndirectDMAOp::verifyGather(
    MemRefType operand_ty, ArrayRef<int64_t> offsets_shape,
    MemRefType result_ty) {
  // We've already thrown an error if the target is not VMEM. so this is just a
  // sanity check.
  CHECK(HasMemorySpace(result_ty, MemorySpace::kVmem));
  uint64_t offsets_rank = offsets_shape.size();
  // Slice [o0, .., on] out of [o0, .., on, s0, .., sm].
  ArrayRef<int64_t> result_offset_dims =
      result_ty.getShape().take_front(offsets_rank);
  // Slice [s0, .., sm] out of [o0, .., on, s0, .., sm].
  ArrayRef<int64_t> result_slice_dims =
      result_ty.getShape().drop_front(offsets_rank);
  // Slice [s0, .., sm] out of [z0, .., zn, s0, .., sm].
  ArrayRef<int64_t> operand_slice_dims =
      operand_ty.getShape().drop_front(offsets_rank);
  uint64_t slice_rank = operand_slice_dims.size();

  const std::string result_shape_str =
      absl::StrJoin(result_ty.getShape(), ", ");

  // Make sure that the output shape is such that there is one output slice per
  // offset.
  // offsets shape : [o0, .., on]
  // result shape  : [o'0, .., o'n, s0, .., sm]
  // [o0, .., on] == [o'0, .., o'n]
  if (!absl::c_equal(offsets_shape, result_offset_dims)) {
    return emitOpError("Offsets shape (")
           << absl::StrJoin(offsets_shape, ", ")
           << ") must match the majormost dimensions of the target (gather "
              "result) shape ("
           << result_shape_str << ")";
  }

  // At each offset, we are copying an ND slice of data. Make sure that the
  // slice shape is the same in the operand and the output for the gather, and
  // in the updates and the operand for the scatter.
  // Operand shape : [z0, .., zn, s0, .., sm]
  // Result shape :  [o0, .., on, s'0, .., s'm]
  // [s0, .., sm] == [s'0, .., s'm]
  if (!absl::c_equal(operand_slice_dims, result_slice_dims)) {
    const std::string plural = slice_rank == 1 ? "" : "s";
    return emitOpError(absl::StrFormat(
        "%d minormost dimension%s of the source (gather operand) shape (%s) "
        "must match the minormost dimension%s of the target (gather result) "
        "shape (%s)",
        slice_rank, plural, absl::StrJoin(operand_ty.getShape(), ", "), plural,
        result_shape_str));
  }
  return success();
}

LogicalResult EnqueueIndirectDMAOp::verifyScatter(
    MemRefType updates_ty, ArrayRef<int64_t> offsets_shape,
    MemRefType operand_ty) {
  // We've already thrown an error if the source is not VMEM. so this is just a
  // sanity check.
  CHECK(HasMemorySpace(updates_ty, MemorySpace::kVmem));
  uint64_t offsets_rank = offsets_shape.size();
  // Slice [o0, .., on] out of [o0, .., on, s0, .., sm].
  ArrayRef<int64_t> updates_offset_dims =
      updates_ty.getShape().take_front(offsets_rank);
  // Slice [s0, .., sm] out of [o0, .., on, s0, .., sm].
  ArrayRef<int64_t> updates_slice_dims =
      updates_ty.getShape().drop_front(offsets_rank);
  // Slice [s0, .., sm] out of [z0, .., zn, s0, .., sm].
  ArrayRef<int64_t> operand_slice_dims =
      operand_ty.getShape().drop_front(offsets_rank);
  uint64_t slice_rank = operand_slice_dims.size();

  const std::string updates_shape_str =
      absl::StrJoin(updates_ty.getShape(), ", ");

  // Make sure that there is one slice of updates per offset
  // offsets shape : [o0, .., on]
  // updates shape : [o'0, .., o'n, s0, .., sm]
  // [o0, .., on] == [o'0, .., o'n]
  if (!absl::c_equal(offsets_shape, updates_offset_dims)) {
    return emitOpError("Offsets shape (")
           << absl::StrJoin(offsets_shape, ", ")
           << ") must match the majormost dimensions of the source "
              "(scatter updates) shape ("
           << updates_shape_str << ")";
  }

  // At each offset, we are copying an ND slice of data. Make sure that the
  // slice shape is the same in the operand and the output for the gather, and
  // in the updates and the operand for the scatter.
  // Updates shape : [o0, .., on, s0, .., sm]
  // Operand shape : [z0, .., zn, s'0, .., s'm]
  // [s0, .., sm] == [s'0, .., s'm]
  if (!absl::c_equal(operand_slice_dims, updates_slice_dims)) {
    const std::string plural = slice_rank == 1 ? "" : "s";
    return emitOpError(absl::StrFormat(
        "%d minormost dimension%s of the source (scatter updates) shape (%s) "
        "must match the minormost dimension%s of the target (scatter operand) "
        "shape (%s)",
        slice_rank, plural, updates_shape_str, plural,
        absl::StrJoin(operand_ty.getShape(), ", ")));
  }
  return success();
}

namespace {
bool HasHbmOrVmemSharedMemorySpace(MemRefType ty) {
  return HasMemorySpace(ty, MemorySpace::kHbm) ||
         HasMemorySpace(ty, MemorySpace::kVmemShared);
}
}  // namespace

FailureOr<bool> EnqueueIndirectDMAOp::isGather() {
  const MemRefType source_ty = getMemRefType(getSource());
  const MemRefType target_ty = getMemRefType(getTarget());
  if (HasHbmOrVmemSharedMemorySpace(source_ty) &&
      HasMemorySpace(target_ty, MemorySpace::kVmem)) {
    return true;
  }
  if (HasMemorySpace(source_ty, MemorySpace::kVmem) &&
      HasHbmOrVmemSharedMemorySpace(target_ty)) {
    return false;
  }
  return emitOpError(
      "The transfer must be between HBM and VMEM, or between VMEM_SHARED and "
      "VMEM");
}

LogicalResult EnqueueIndirectDMAOp::verify() {
  FailureOr<CoreType> issuing_core = GetCoreTypeOfParentFunc(**this);
  if (failed(issuing_core)) {
    return issuing_core;
  }
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
    return verifyGather(/*operand_ty=*/source_ty,
                        /*offsets_shape=*/offsets_shape,
                        /*result_ty=*/target_ty);
  }
  return verifyScatter(/*updates_ty=*/source_ty,
                       /*offsets_shape=*/offsets_shape,
                       /*operand_ty=*/target_ty);
}

// TODO(b/395630795): Remove after 2025-08-10.
LogicalResult WaitDMAOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    emitOpError("DMA wait semaphore must be rank 0");
    return failure();
  }
  return success();
}

void WaitDMA2Op::build(OpBuilder &builder, OperationState &state,
                       Value semaphore, Value src, Value dst) {
  build(builder, state, semaphore, src, dst, /*device_id=*/nullptr,
        /*core_id=*/nullptr);
}

LogicalResult WaitDMA2Op::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    emitOpError("DMA wait semaphore must be rank 0");
    return failure();
  }
  return success();
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
                                           PatternRewriter &rewriter) {
  bool can_convert_to_simple_load = true;
  for (int i = 0; i < op.getSublaneOffsets().size(); ++i) {
    if (op.getSublaneOffsets()[i] != i) {
      can_convert_to_simple_load = false;
      break;
    };
  }
  if (can_convert_to_simple_load) {
    rewriter.replaceOpWithNewOp<tpu::LoadOp>(
        op, op.getType(), op.getBase(), op.getIndices(), op.getSublaneMask(),
        /*sublane_stride=*/nullptr);
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
                                            PatternRewriter &rewriter) {
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
                                              /*mask=*/nullptr,
                                              /*sublane_stride=*/nullptr);
  }
  return success();
}

LogicalResult FPToSIOp::canonicalize(FPToSIOp op, PatternRewriter &rewriter) {
  if (auto round_op = op.getInput().getDefiningOp<mlir::math::RoundEvenOp>()) {
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

LogicalResult LogOp::verify() {
  FailureOr<CoreType> logging_core = GetCoreTypeOfParentFunc(**this);
  if (failed(logging_core)) {
    return logging_core;
  }
  bool is_sc_core = *logging_core == CoreType::kScScalarSubcore ||
                    *logging_core == CoreType::kScVectorSubcore;
  if (is_sc_core && getFormattedAttr() != nullptr &&
      getFormattedAttr().getValue()) {
    return emitOpError("Formatted logging is not supported on SC");
  }
  if (is_sc_core && getInputs().size() > 1) {
    return emitOpError("SC logging only supports 0 or 1 inputs");
  }
  if (is_sc_core && getInputs().size() == 1) {
    Type input_type = getInputs().front().getType();
    if (!llvm::isa<MemRefType, IntegerType, FloatType, IndexType>(input_type)) {
      return emitOpError("SC logging only supports memrefs or scalars");
    }
  }
  switch (*logging_core) {
    case CoreType::kTc:
    case CoreType::kScScalarSubcore:
      return success();
    case CoreType::kScVectorSubcore:
      return emitOpError("Log op is not supported on the SC vector subcore");
  }
  return emitOpError(absl::StrFormat("Unexpected core type: %s",
                                     stringifyCoreType(*logging_core)));
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

void PackSubelementsOp::build(OpBuilder &builder, OperationState &state,
                              const VectorType output_type,
                              const ArrayRef<Value> padded_sources,
                              const PackFormat pack_format) {
  SmallVector<Value> sources;
  SmallVector<int32_t> positions;
  for (size_t i = 0; i < padded_sources.size(); ++i) {
    if (padded_sources[i] != nullptr) {
      sources.push_back(padded_sources[i]);
      positions.push_back(i);
    }
  }
  build(builder, state, output_type, sources, positions, pack_format);
}

SmallVector<Value> PackSubelementsOp::getPaddedSources(
    ValueRange sources, const ArrayRef<int32_t> positions,
    const int packing_factor) {
  SmallVector<Value> padded_sources(packing_factor);
  for (const auto [source, position] : llvm::zip(sources, positions)) {
    padded_sources[position] = source;
  }
  return padded_sources;
}

LogicalResult PackSubelementsOp::verify() {
  if (getSources().empty()) {
    return emitOpError("At least one source is required");
  }
  if (getPositions().size() != getSources().size()) {
    return emitOpError("Size of sources and positions must match");
  }
  const int packing_factor = cast<VectorType>(getSources().front().getType())
                                 .getElementTypeBitWidth() /
                             getType().getElementTypeBitWidth();
  SmallVector<bool> seen_positions(packing_factor, false);
  for (const int32_t position : getPositions()) {
    if (position < 0 || packing_factor <= position) {
      return emitOpError("Positions must be between 0 and the packing factor");
    }
    if (seen_positions[position]) {
      return emitOpError("Positions must be unique");
    }
    seen_positions[position] = true;
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
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  VectorType source_vty = cast<VectorType>(operands[0].getType());
  VectorType indices_vty = cast<VectorType>(operands[1].getType());
  inferredReturnTypes.push_back(
      VectorType::get(indices_vty.getShape(), source_vty.getElementType()));
  return success();
}

LogicalResult AssumeMultipleOp::verify() {
  auto operand_value = getValue();
  auto divisor = getMultiple();
  if (auto cst_op = operand_value.getDefiningOp<arith::ConstantOp>()) {
    auto int_attr = dyn_cast<IntegerAttr>(cst_op.getValue());
    // Illegal usage of AssumeMultipleOp.
    if (!int_attr) {
      return emitOpError(
                 "Illegal user annotation, expected an integer, but got ")
             << cst_op.getValue();
    }
    if (int_attr.getInt() % divisor != 0) {
      return emitOpError(
                 "Illegal user annotation, expected an integer that is "
                 "divisible by the multiple, but got ")
             << int_attr.getInt() << " % " << divisor;
    }
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
  const llvm::fltSemantics &targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr, FloatAttr::ValueType,
                         FloatAttr::ValueType, /*PoisonAttr=*/void>(
      adaptor.getOperands(), getType(),
      [this, &targetSemantics](const APFloat &a, bool &castStatus) {
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
  const llvm::fltSemantics &targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr, FloatAttr::ValueType,
                         FloatAttr::ValueType, /*PoisonAttr=*/void>(
      adaptor.getOperands(), getType(),
      [&targetSemantics](const APFloat &a, bool &castStatus) {
        FailureOr<APFloat> result = convertFloatValue(a, targetSemantics);
        if (failed(result)) {
          castStatus = false;
          return a;
        }
        return *result;
      });
}

}  // namespace tpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_ops.cc.inc"

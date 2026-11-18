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

#ifndef XLA_MOSAIC_DIALECT_TPU_TPU_OPS_IMPL_H_
#define XLA_MOSAIC_DIALECT_TPU_TPU_OPS_IMPL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/util.h"

namespace mlir::tpu::impl {

LogicalResult verifyGather(Operation* op, ArrayRef<int64_t> operand_shape,
                           ArrayRef<int64_t> offsets_shape,
                           ArrayRef<int64_t> result_shape);

LogicalResult verifyScatter(Operation* op, ArrayRef<int64_t> updates_shape,
                            ArrayRef<int64_t> offsets_shape,
                            ArrayRef<int64_t> operand_shape);

template <typename MemorySpaceT, typename CoreTypeT>
bool hasSharedMemorySpace(MemorySpaceT memory_space,
                          std::optional<CoreTypeT> core_type) {
  return memory_space == MemorySpaceT::kHbm ||
         (memory_space == MemorySpaceT::kVmem && core_type.has_value() &&
          *core_type == CoreTypeT::kTc) ||
         memory_space == MemorySpaceT::kVmemShared;
}

// True if source represents a shared memory space and target represents a local
// memory space. TC VMEM is "shared" here but it is not shared between TCs.
template <typename MemorySpaceT, typename CoreTypeT>
FailureOr<bool> isGather(Operation& op, MemorySpaceT source_memory_space,
                         std::optional<CoreTypeT> source_core_type,
                         MemorySpaceT target_memory_space,
                         std::optional<CoreTypeT> target_core_type) {
  if (impl::hasSharedMemorySpace(source_memory_space, source_core_type) &&
      target_memory_space == MemorySpaceT::kVmem) {
    return true;
  }
  if (source_memory_space == MemorySpaceT::kVmem &&
      impl::hasSharedMemorySpace(target_memory_space, target_core_type)) {
    return false;
  }
  return op.emitOpError(
      "The transfer must be between HBM and VMEM, VMEM_SHARED and VMEM, or TC "
      "VMEM and VMEM");
}

// Op verifiers and return type inference, parameterized by `DialectBindings`,
// which supplies the types and hooks they depend on as static members (see
// `TpuDialectBindings` in tpu_ops.cc):
//   - Type aliases: `DMASemaphoreType`, `EnqueueIndirectDMAOp`, `MemorySpace`,
//     `MemorySpaceAttr`, `PackSubelementsOp`, `ReductionKind`,
//     `SemaphoreType`, `TiledLayoutAttr` and `WaitIndirectDMAOp`.
//   - Static hooks: `getTypeBitwidth`, `getElementTypeBitwidth`,
//     `getMemRefType` and `HasMemorySpace`.
// `MemorySpace` must provide `kAny`, `kHbm`, `kSmem`, `kVmem` and
// `kVmemShared`.
template <typename DialectBindings>
struct OpVerifiers : private DialectBindings {
  using typename DialectBindings::DMASemaphoreType;
  using typename DialectBindings::EnqueueIndirectDMAOp;
  using typename DialectBindings::MemorySpace;
  using typename DialectBindings::MemorySpaceAttr;
  using typename DialectBindings::PackSubelementsOp;
  using typename DialectBindings::ReductionKind;
  using typename DialectBindings::SemaphoreType;
  using typename DialectBindings::TiledLayoutAttr;
  using typename DialectBindings::WaitIndirectDMAOp;

  using DialectBindings::getElementTypeBitwidth;
  using DialectBindings::getMemRefType;
  using DialectBindings::getTypeBitwidth;
  using DialectBindings::HasMemorySpace;

  template <typename OpTy>
  static LogicalResult verifyPackOp(OpTy op) {
    if (op.getSources().empty()) {
      return op.emitOpError("At least one source is required");
    }
    const VectorType source_type =
        cast<VectorType>(op.getSources().front().getType());
    const VectorType result_type = op.getResult().getType();
    if (!llvm::all_of(op.getSources(), [&](Value source) {
          return source.getType() == source_type;
        })) {
      return op.emitOpError("All sources must have the same type");
    }
    if (op.getPositions().size() != op.getSources().size()) {
      return op.emitOpError("Size of sources and positions must match");
    }
    int64_t source_packing;
    if (source_type.getRank() == result_type.getRank()) {
      source_packing = source_type.getShape().back();
    } else if (source_type.getRank() + 1 == result_type.getRank()) {
      source_packing = 1;
    } else {
      return op.emitOpError(
          "The source rank must be equal to the result rank, or smaller by 1");
    }
    if (result_type.getShape().drop_back() !=
        source_type.getShape().take_front(result_type.getRank() - 1)) {
      return op.emitOpError(
          "Source and result shapes must match except for packing (result's "
          "minormost) dimension");
    }
    const int64_t result_packing = result_type.getShape().back();

    if (result_packing % source_packing != 0) {
      return op.emitOpError(
          "Result packing must be a multiple of source packing");
    }
    const int64_t ratio = result_packing / source_packing;

    if constexpr (std::is_same_v<OpTy, PackSubelementsOp>) {
      const int source_bitwidth = getElementTypeBitwidth(source_type);
      const int result_bitwidth = getElementTypeBitwidth(result_type);
      if (source_bitwidth % result_bitwidth != 0 ||
          source_bitwidth / result_bitwidth != ratio) {
        return op.emitOpError(
            "Ratio of bitwidths must match the ratio of packing dimensions");
      }
    }

    SmallVector<bool> seen_positions(ratio, false);
    for (const int32_t position : op.getPositions()) {
      if (position < 0 || ratio <= position) {
        return op.emitOpError(
                   "Positions must be between 0 and the packing ratio (")
               << ratio << "), got " << position;
      }
      if (seen_positions[position]) {
        return op.emitOpError("Positions must be unique");
      }
      seen_positions[position] = true;
    }
    CHECK_LE(op.getSources().size(), ratio);
    return success();
  }

  template <typename BitcastOpT>
  static LogicalResult verifyBitcastOp(BitcastOpT op) {
    auto in_ty = op.getInput().getType();
    auto out_ty = op.getOutput().getType();
    auto in_bitwidth = getElementTypeBitwidth(in_ty);
    auto out_bitwidth = getElementTypeBitwidth(out_ty);
    if (in_bitwidth != out_bitwidth) {
      if (in_ty.getRank() < 2 || out_ty.getRank() < 2) {
        return op.emitError(
            "Not implemented: bitcast between different bitwidths on a 1D "
            "vector.");
      }
      SmallVector<int64_t, 4> in_shape(in_ty.getShape());
      SmallVector<int64_t, 4> out_shape(out_ty.getShape());
      *(in_shape.end() - 2) *= in_bitwidth;
      *(out_shape.end() - 2) *= out_bitwidth;
      if (in_shape != out_shape) {
        return op.emitError(
            "Expected input and output shapes are the same after multiplying "
            "the second-minor dimension by the ratio of bitwidths.");
      }
    } else if (in_ty.getShape() != out_ty.getShape()) {
      return op.emitError(
          "Expected input and output shapes are the same when bitwidth does "
          "not change.");
    }
    return success();
  }

  template <typename MemRefSliceOpT>
  static LogicalResult verifyMemRefSliceOp(MemRefSliceOpT op) {
    auto source_type = op.getMemRef().getType();
    auto target_type = op.getType();
    auto source_layout = source_type.getLayout();
    auto target_layout = target_type.getLayout();
    auto target_memory_space = target_type.getMemorySpace();
    auto indices = op.getBaseIdx();
    auto slice_shape = op.getResult().getType().getShape();
    if (op.getDynamicSizes().size() != target_type.getNumDynamicDims()) {
      return op.emitOpError(
          "Number of provided dynamic dimensions sizes must match the number "
          "of dynamic dimensions in the target type.");
    }
    auto source_shape = source_type.getShape();
    bool is_semaphore = HasMemorySpace(source_type, MemorySpace::kSemaphoreMem);
    if (is_semaphore &&
        !isa<SemaphoreType, DMASemaphoreType>(source_type.getElementType())) {
      return op.emitOpError(
          "References to semaphore memory space must have a semaphore element "
          "type.");
    }
    if (indices.size() != slice_shape.size() ||
        indices.size() != source_shape.size()) {
      return op.emitOpError("Indices and slice shapes must match.");
    }
    // TODO(apaszke): Check that the result has a smaller shape.
    // TODO(apaszke): Check that strides are equivalent.
    if (target_memory_space != source_type.getMemorySpace()) {
      return op.emitOpError("Memory spaces do not match.");
    }
    if (isa<TiledLayoutAttr>(source_layout) !=
        isa<TiledLayoutAttr>(target_layout)) {
      return op.emitOpError("Source and target layouts must match.");
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
      return op.emitOpError("Source memref has ")
             << rank << " dimensions, but the tiling has " << tile_rank
             << " dimensions. The memref must have at least as many dimensions "
                "as the tiling.";
    }
    return success();
  }

  template <typename SharedMemRefSliceOpT>
  static LogicalResult verifySharedMemRefSliceOp(
      SharedMemRefSliceOpT op, MemorySpaceAttr expected_target_memory_space) {
    auto source_type = op.getMemRef().getType();
    auto target_type = op.getType();
    auto source_memory_space = source_type.getMemorySpace();
    auto target_memory_space = target_type.getMemorySpace();

    auto expected_source_memory_space =
        MemorySpaceAttr::get(op.getContext(), MemorySpace::kVmemShared);
    if (source_memory_space != expected_source_memory_space) {
      return op.emitOpError("Source memref must have memory space ")
             << expected_source_memory_space;
    }
    if (target_memory_space != expected_target_memory_space &&
        target_memory_space !=
            MemorySpaceAttr::get(op.getContext(), MemorySpace::kVmem)) {
      return op.emitOpError("Target memref must have memory space ")
             << expected_target_memory_space;
    }
    if (source_type.getElementType() != target_type.getElementType()) {
      return op.emitOpError("Source and target element types must match.");
    }

    ArrayRef<int64_t> source_shape = source_type.getShape();
    ArrayRef<int64_t> target_shape = target_type.getShape();
    if (source_shape.size() != target_shape.size()) {
      return op.emitOpError("Target shape rank must match source shape rank.");
    }
    if (source_shape.empty()) {
      return op.emitOpError(
          "Source and target memrefs must be at least rank 1.");
    }
    if (source_shape.drop_back() != target_shape.drop_back()) {
      return op.emitOpError(
          "Target shape must match source shape for all dimensions except the "
          "last.");
    }
    if (source_shape.back() == 0 || target_shape.back() == 0) {
      return op.emitOpError(
          "Source and target minormost dimension must be greater than 0.");
    }
    if (ShapedType::isDynamic(source_shape.back()) ||
        ShapedType::isDynamic(target_shape.back())) {
      return op.emitOpError(
          "Source and target minormost dimension cannot be dynamic.");
    }
    if (source_shape.back() % target_shape.back() != 0) {
      return op.emitOpError(
          "Source shape's minormost dimension must be divisible by target "
          "shape's minormost dimension.");
    }
    return success();
  }

  template <typename MemRefSqueezeOpT>
  static LogicalResult verifyMemRefSqueezeOp(MemRefSqueezeOpT op) {
    MemRefType input_type = op.getInput().getType();
    MemRefType result_type = op.getType();

    if (result_type.getMemorySpace() != input_type.getMemorySpace()) {
      return op.emitOpError("Memory spaces do not match.");
    }

    if (result_type.getElementType() != input_type.getElementType()) {
      return op.emitOpError("Element types don't match.");
    }

    const ArrayRef<int64_t> input_shape = input_type.getShape();
    const ArrayRef<int64_t> result_shape = result_type.getShape();
    // NOTE: In cases where there is flexibility on which dimension to squeeze,
    //   such as 1x1x2 -> 1x2 (can squeeze dimension 1 or 2), this may choose
    //   dimensions that have padding and fail in
    //   inferMemRefSqueezeResultLayout.
    // TODO(tlongeri): Unify logic in inferMemRefReshape such that we don't
    //  need computeSqueezedDimsChecked at all.
    FAILUREOR_ASSIGN_OR_RETURN(
        const SmallVector<int64_t> squeezed,
        mosaic::computeSqueezedDimsChecked(op, input_shape, result_shape));
    if (squeezed.empty() && input_shape != result_shape) {
      return op.emitOpError(
          "Source and target shapes must be the same if no dimensions are "
          "squeezed.");
    }

    if (isa<StridedLayoutAttr>(input_type.getLayout()) ||
        isa<StridedLayoutAttr>(result_type.getLayout())) {
      // TODO(b/568373460): This is for compatibility with older serialized
      //   artifacts outside of the guaranteed compatibility window. Remove once
      //   all artifacts are updated.
      return success();
    }

    FAILUREOR_ASSIGN_OR_RETURN(
        const MemRefLayoutAttrInterface expected_result_layout,
        inferMemRefSqueezeResultLayout(input_type.getLayout(), squeezed,
                                       [&]() { return op.emitOpError(); }));
    if (result_type.getLayout() != expected_result_layout) {
      return op.emitOpError("Expected result layout to be ")
             << expected_result_layout;
    }
    return success();
  }

  static FailureOr<MemRefLayoutAttrInterface> inferMemRefSqueezeResultLayout(
      const MemRefLayoutAttrInterface input_layout,
      const ArrayRef<int64_t> squeezed,
      const function_ref<InFlightDiagnostic()> emit_error) {
    // TODO(tlongeri): Make inferMemRefReshapeResultLayout more general and
    // use that instead.
    MLIRContext* const ctx = input_layout.getContext();
    if (auto tiled_layout = dyn_cast<TiledLayoutAttr>(input_layout)) {
      const int64_t input_rank = tiled_layout.getRank();
      const int64_t result_rank = input_rank - squeezed.size();

      SmallVector<int64_t> tile_strides;
      tile_strides.reserve(result_rank);
      for (int64_t i = 0; i < input_rank; ++i) {
        if (!llvm::is_contained(squeezed, i)) {
          tile_strides.push_back(tiled_layout.getTileStrides()[i]);
        }
      }

      const ArrayRef<xla::Tile> tiles = tiled_layout.getTiles();

      if (tiles.size() == 1 && tiles[0].dimensions().size() == 2 &&
          tiles[0].dimension(0) == 1 &&
          !llvm::is_contained(squeezed, input_rank - 1) &&
          llvm::is_contained(squeezed, input_rank - 2) && result_rank >= 2) {
        // For legacy reasons, for T(1, B) that squeezes the 2nd minor, infer
        // T(1, B) instead of T(B) like in the code below.
        return MemRefLayoutAttrInterface(TiledLayoutAttr::get(
            ctx, {xla::Tile({1, tiles[0].dimension(1)})}, tile_strides));
      }

      SmallVector<xla::Tile> result_tiles;
      result_tiles.reserve(tiles.size());
      // Perform expansion as in getExpandedShape, maintaining a mapping from
      // expanded shape dimension to corresponding input dimension.
      SmallVector<int64_t> expanded_dims = llvm::to_vector(
          llvm::iota_range<int64_t>(0, input_rank, /*Inclusive=*/false));
      for (const xla::Tile& input_tile : tiles) {
        const int64_t tile_rank = input_tile.dimensions().size();
        const int64_t expanded_rank = expanded_dims.size();
        SmallVector<int64_t> tile;
        for (int64_t i = 0; i < tile_rank; ++i) {
          const int64_t input_dim =
              expanded_dims[expanded_rank - tile_rank + i];
          if (llvm::is_contained(squeezed, input_dim)) {
            if (input_tile.dimension(i) != 1) {
              return emit_error() << "Dimension " << input_dim
                                  << " is padded but is squeezed.";
            }
          } else {
            tile.push_back(input_tile.dimension(i));
          }
          expanded_dims.push_back(input_dim);
        }
        if (!tile.empty()) {
          result_tiles.emplace_back(tile);
        }
      }
      return MemRefLayoutAttrInterface(
          TiledLayoutAttr::get(ctx, result_tiles, tile_strides));
    }
    if (auto affine_map_attr = dyn_cast<AffineMapAttr>(input_layout);
        affine_map_attr && affine_map_attr.isIdentity()) {
      const int64_t input_rank = affine_map_attr.getValue().getNumInputs();
      const int64_t result_rank = input_rank - squeezed.size();
      // Untiled squeeze
      return MemRefLayoutAttrInterface(AffineMapAttr::get(
          AffineMap::getMultiDimIdentityMap(result_rank, ctx)));
    }
    return emit_error() << "Only tiled or identity layouts supported.";
  }

  template <typename MemRefReshapeOpT>
  static LogicalResult verifyMemRefReshapeOp(MemRefReshapeOpT op) {
    const MemRefType input_ty = op.getInput().getType();
    const MemRefType result_ty = op.getType();
    if (result_ty.getMemorySpace() != input_ty.getMemorySpace()) {
      return op.emitOpError("Memory spaces do not match.");
    }
    if (result_ty.getElementType() != input_ty.getElementType()) {
      return op.emitOpError("Element types don't match.");
    }
    int64_t input_static_elements = 1;
    int64_t result_static_elements = 1;
    int64_t input_dynamic_dims = 0;
    int64_t result_dynamic_dims = 0;
    for (const int64_t dim : input_ty.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        ++input_dynamic_dims;
      } else {
        input_static_elements *= dim;
      }
    }
    for (const int64_t dim : result_ty.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        ++result_dynamic_dims;
      } else {
        result_static_elements *= dim;
      }
    }
    const bool is_static = input_dynamic_dims == 0;
    if (is_static != (result_dynamic_dims == 0)) {
      return op.emitOpError(
          "Input and result shapes must be both static or both dynamic.");
    }
    if (is_static && input_static_elements != result_static_elements) {
      return op.emitOpError("Input and result number of elements don't match.");
    }
    if (result_dynamic_dims > 1) {
      return op.emitOpError(
          "Multiple result dynamic dimensions (their values are ambiguous).");
    }
    // The above checks are fundamental, the following is simply unimplemented:
    if (!is_static && (input_dynamic_dims > 1 ||
                       input_static_elements != result_static_elements)) {
      // inferMemRefReshapeResultLayout relies on this check. Make sure to
      // update it when removing this.
      return op.emitOpError(
          "Not implemented: Input with multiple dynamic dimensions or a "
          "changing single dynamic dimension.");
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const MemRefLayoutAttrInterface expected_result_layout,
        inferMemRefReshapeResultLayout(input_ty, result_ty.getShape(),
                                       [&]() { return op.emitOpError(); }));
    if (expected_result_layout != result_ty.getLayout()) {
      return op.emitOpError("Expected result layout to be ")
             << expected_result_layout;
    }
    return success();
  }

  template <typename MemRefPadOpT>
  static LogicalResult verifyMemRefPadOp(MemRefPadOpT op) {
    const MemRefType input_ty = op.getInput().getType();
    const MemRefType result_ty = op.getType();
    if (result_ty.getMemorySpace() != input_ty.getMemorySpace()) {
      return op.emitOpError("Memory spaces do not match.");
    }
    if (result_ty.getElementType() != input_ty.getElementType()) {
      return op.emitOpError("Element types don't match.");
    }
    if (result_ty.getRank() != input_ty.getRank()) {
      return op.emitOpError("Input and result ranks must match.");
    }
    if (result_ty.getLayout() != input_ty.getLayout()) {
      return op.emitOpError("Input and result layouts must match.");
    }

    auto tiled_layout = dyn_cast<TiledLayoutAttr>(input_ty.getLayout());
    if (!tiled_layout && !input_ty.getLayout().isIdentity()) {
      return op.emitOpError("Only tiled or identity layouts supported.");
    }
    ArrayRef<int64_t> first_tile;
    if (tiled_layout && !tiled_layout.getTiles().empty()) {
      first_tile = tiled_layout.getTiles().front().dimensions();
    }
    const int64_t first_tiled_dim = input_ty.getRank() - first_tile.size();
    for (int64_t i = 0; i < input_ty.getRank(); ++i) {
      const int64_t in_dim = input_ty.getDimSize(i);
      const int64_t res_dim = result_ty.getDimSize(i);
      if (in_dim == res_dim) {
        continue;
      }
      if (ShapedType::isDynamic(in_dim) || ShapedType::isDynamic(res_dim)) {
        return op.emitOpError("Cannot pad dynamic dimension ") << i << ".";
      }
      if (res_dim < in_dim) {
        return op.emitOpError("Result dimension ")
               << i << " (" << res_dim
               << ") must be greater than or equal to input dimension ("
               << in_dim << ").";
      }
      if (tiled_layout) {
        const int64_t tile_size =
            i < first_tiled_dim ? 1 : first_tile[i - first_tiled_dim];
        const int64_t aligned_dim = llvm::alignTo(in_dim, tile_size);
        if (res_dim > aligned_dim) {
          return op.emitOpError("Result dimension ")
                 << i << " (" << res_dim
                 << ") must not exceed the tile-aligned bound (" << aligned_dim
                 << ") of input dimension " << in_dim << " for tile size "
                 << tile_size << ".";
        }
      }
    }
    return success();
  }

  static FailureOr<MemRefLayoutAttrInterface> inferMemRefReshapeResultLayout(
      const MemRefType input_type, const ArrayRef<int64_t> result_shape,
      function_ref<InFlightDiagnostic()> emit_error) {
    MLIRContext* const ctx = input_type.getContext();
    const ArrayRef<int64_t> input_shape = input_type.getShape();
    const MemRefLayoutAttrInterface input_layout = input_type.getLayout();
    const int64_t input_rank = input_type.getRank();
    const int64_t result_rank = result_shape.size();
    if (const auto input_tiled_layout =
            dyn_cast<TiledLayoutAttr>(input_layout)) {
      if (!input_tiled_layout.tilesAreKnownContiguous(input_type.getShape())) {
        return emit_error()
               << "Not implemented: Reshape on a non-contiguous memref.";
      }
      const ArrayRef<xla::Tile> input_tiles = input_tiled_layout.getTiles();
      if (input_tiles.empty()) {
        // Untiled reshape
        return MemRefLayoutAttrInterface(
            TiledLayoutAttr::getContiguous(ctx, /*tiles=*/{}, result_shape));
      }
      const ArrayRef<int64_t> input_first_tile =
          input_tiles.front().dimensions();
      const int64_t input_first_tile_rank = input_first_tile.size();
      const int64_t input_untiled_rank = input_rank - input_first_tile_rank;

      const TiledLayoutAttr preserved_tile_layout = [&]() -> TiledLayoutAttr {
        if (result_rank < input_first_tile_rank) {
          return nullptr;
        }
        int64_t i = 0;
        // Leading tile dimensions of size 1 can be effectively ignored.
        while (i < input_first_tile_rank && input_first_tile[i] == 1) {
          ++i;
        }
        if (i == input_first_tile_rank) {
          // Untiled reshape
          return TiledLayoutAttr::getContiguous(ctx, /*tiles=*/{},
                                                result_shape);
        }
        // Excluding leading ones, tiled dimensions must remain unchanged, with
        // the possible exception of the first. Dimensions can be folded into or
        // unfolded from it if it is tile-aligned in both the input and result.
        // NOTE: This relies on the verifier enforcing that, if there is a
        //       dynamic dimension, it is preserved between input and result.
        auto input_dim_it = input_shape.end() - input_first_tile_rank + i;
        auto result_dim_it = result_shape.end() - input_first_tile_rank + i;
        bool can_preserve_tiling = *input_dim_it == *result_dim_it ||
                                   (ShapedType::isStatic(*input_dim_it) &&
                                    ShapedType::isStatic(*result_dim_it) &&
                                    *input_dim_it % input_first_tile[i] == 0 &&
                                    *result_dim_it % input_first_tile[i] == 0);
        ++i;
        ++input_dim_it;
        ++result_dim_it;
        for (; i < input_first_tile_rank;
             ++i, ++input_dim_it, ++result_dim_it) {
          can_preserve_tiling &= *input_dim_it == *result_dim_it;
        }
        if (!can_preserve_tiling) {
          return nullptr;
        }
        return TiledLayoutAttr::getContiguous(ctx, input_tiles, result_shape);
      }();
      if (preserved_tile_layout != nullptr) {
        return MemRefLayoutAttrInterface(preserved_tile_layout);
      }
      const TiledLayoutAttr reshaped_tile_layout = [&]() -> TiledLayoutAttr {
        if (input_tiles.size() != 1) {
          return nullptr;
        }
        // To reason about the reshape, it can be helpful to conceptually view
        // it as having the following steps:
        //
        // 1. Expand the unpadded tiled dimensions as follows:
        //      16x256 with T(8, 128) -> 2x8x2x128
        //    Note that this is not the same as getExpandedShape. Expanded
        //    dimensions from the same unexpanded dimension are kept adjacent.
        //
        // 2. Collapse adjacent dimensions that have compatible striding.
        //    Note that tile dimensions can be collapsed together iff the
        //    non-tile dimensions that separate them are of size 1 (i.e. the
        //    original unexpanded dimension is exactly one tile).
        //
        // 3. Expand the collapsed dimensions into the result shape. Fail if
        //    this is not possible.
        //
        // The code below does not implement this _exactly_, to maintain a tiled
        // representation, but performs a similar process. The expanded shape
        // can be viewed as alternating tile and non-tile dimensions. Like
        // above, tile dimensions can be collapsed together iff the non-tile
        // dimensions that separate them are of size 1. Likewise, non-tile
        // dimensions can be collapsed together iff the tile dimensions that
        // separate them are of size 1. We iteratively collapse tile and
        // non-tile dimensions and then try expanding them into the result
        // shape.
        auto get_input_tile_size = [&](int64_t input_dim) {
          return input_untiled_rank <= input_dim
                     ? input_first_tile[input_dim - input_untiled_rank]
                     : 1;
        };
        // Dynamic and padded dimensions must be preserved exactly.
        // NOTE: This relies on the verifier enforcing that, if there is a
        //       dynamic dimension, it is preserved between input and result.
        auto input_dim_is_dynamic_or_padded = [&](int64_t input_dim) {
          const int64_t input_tile_size = get_input_tile_size(input_dim);
          return ShapedType::isDynamic(input_shape[input_dim]) ||
                 input_shape[input_dim] % input_tile_size != 0;
        };
        int64_t input_dim = 0;
        int64_t result_dim = 0;
        SmallVector<int64_t> result_tile;
        while (input_dim < input_rank || result_dim < result_rank) {
          if (input_dim < input_rank &&
              input_dim_is_dynamic_or_padded(input_dim)) {
            if (result_dim == result_rank ||
                input_shape[input_dim] != result_shape[result_dim]) {
              return nullptr;
            }
            if (input_untiled_rank <= input_dim) {
              result_tile.push_back(
                  input_first_tile[input_dim - input_untiled_rank]);
            }
            ++input_dim;
            ++result_dim;
            continue;
          }
          // Take as many non-tile dimensions as possible
          int64_t non_tile_elements = 1;
          while (input_dim < input_rank) {
            if (input_dim_is_dynamic_or_padded(input_dim)) {
              break;
            }
            const int64_t input_tile_size = get_input_tile_size(input_dim);
            non_tile_elements *= input_shape[input_dim] / input_tile_size;
            if (input_tile_size != 1) {
              break;
            }
            ++input_dim;
          }
          // Take as many tile dimensions as possible.
          // TODO(tlongeri): This could be generalized to multiple tiles, if
          // they do not tile across previous tiles (i.e. they are equal to the
          // previous tile's unit subtile), by iteratively taking dims from the
          // most nested tiles up to the least nested one.
          int64_t tile_elements = 1;
          while (input_dim < input_rank) {
            if (input_dim_is_dynamic_or_padded(input_dim)) {
              break;
            }
            tile_elements *= get_input_tile_size(input_dim);
            ++input_dim;
            if (input_dim == input_rank ||
                input_shape[input_dim] != get_input_tile_size(input_dim)) {
              break;
            }
          }

          // Expand non-tile dimensions
          while (result_dim < result_rank) {
            CHECK_LT(result_dim, result_rank);
            if (ShapedType::isDynamic(result_shape[result_dim])) {
              break;
            }
            if (non_tile_elements % result_shape[result_dim] == 0) {
              if (!result_tile.empty()) {
                result_tile.push_back(1);
              }
              non_tile_elements /= result_shape[result_dim];
              ++result_dim;
            } else if (result_shape[result_dim] % non_tile_elements == 0) {
              // non_tile_elements is left holding the non-tile size for the
              // current result dimension.
              break;
            } else {
              // Cannot expand into the desired shape
              return nullptr;
            }
          }
          // Expand tile dimensions
          while (result_dim < result_rank) {
            CHECK_LT(result_dim, result_rank);
            if (ShapedType::isDynamic(result_shape[result_dim])) {
              break;
            }
            CHECK_EQ(result_shape[result_dim] % non_tile_elements, 0);
            const int64_t result_dim_size =
                result_shape[result_dim] / non_tile_elements;
            non_tile_elements = 1;
            if (tile_elements % result_dim_size == 0) {
              if (!result_tile.empty() || result_dim_size != 1) {
                result_tile.push_back(result_dim_size);
              }
              tile_elements /= result_dim_size;
              ++result_dim;
            } else if (result_dim_size % tile_elements == 0) {
              break;
            } else {
              // Cannot expand into the desired shape
              return nullptr;
            }
          }
          if (non_tile_elements != 1 || tile_elements != 1) {
            // Cannot expand into the desired shape
            return nullptr;
          }
        }
        return TiledLayoutAttr::getContiguous(
            ctx, /*tiles=*/{xla::Tile(result_tile)}, result_shape);
      }();
      if (reshaped_tile_layout != nullptr) {
        return MemRefLayoutAttrInterface(reshaped_tile_layout);
      }
      // TODO(tlongeri): Unify preserved_tile_layout, reshaped_tile_layout cases
      //                 by generalizing reshaped_tile_layout case to multiple
      //                 tiles.
      return emit_error()
             << "Not implemented: Supported only when (a) tiled dimensions "
                "remain unchanged, except maybe for tile-aligned "
                "folding/unfolding of the first tiled dimension, or (b) the "
                "input layout has a single tile, under specific conditions.";
    }
    if (input_layout.isIdentity()) {
      // Untiled reshape
      return MemRefLayoutAttrInterface(AffineMapAttr::get(
          AffineMap::getMultiDimIdentityMap(result_rank, ctx)));
    }
    return emit_error() << "Only tiled or identity layouts supported.";
  }

  template <typename TransposeOpT>
  static LogicalResult verifyTransposeOp(TransposeOpT op) {
    auto source_type = op.getSourceVectorType();
    auto permutation = op.getPermutation();
    auto output_type = op.getResultVectorType();
    auto input_shape = source_type.getShape();
    auto output_shape = output_type.getShape();
    if (source_type.getElementType() != output_type.getElementType()) {
      return op.emitOpError("Expected input and output element types to match");
    }
    if (permutation.size() != source_type.getRank()) {
      return op.emitOpError("Expected permutation rank to match input rank");
    }
    if (permutation.size() != output_type.getRank()) {
      return op.emitOpError("Expected permutation rank to match output rank");
    }
    std::vector<bool> seen_dims(source_type.getRank(), false);
    for (int64_t dim : permutation) {
      if (dim < 0 || dim >= source_type.getRank()) {
        return op.emitOpError("Permutation element out of bounds: ") << dim;
      }
      if (seen_dims[dim]) {
        return op.emitOpError("Permutation element repeated: ") << dim;
      }
      seen_dims[dim] = true;
    }
    for (int i = 0; i < source_type.getRank(); ++i) {
      if (input_shape[permutation[i]] != output_shape[i]) {
        return op.emitOpError(
            "Expected input shape permuted by the given permutation to match "
            "the output shape");
      }
    }
    return success();
  }

  template <typename MemRefBitcastOpT>
  static LogicalResult verifyMemRefBitcastOp(MemRefBitcastOpT op) {
    auto src_ty = op.getInput().getType();
    auto tgt_ty = op.getType();
    FAILUREOR_ASSIGN_OR_RETURN(
        auto result_type,
        inferMemRefBitcastResultType(src_ty, tgt_ty.getElementType(),
                                     [&]() { return op.emitOpError(); }));
    if (result_type != tgt_ty) {
      return op.emitOpError("Expected result type to be ") << result_type;
    }
    return success();
  }

  static mlir::FailureOr<MemRefType> inferMemRefBitcastResultType(
      const MemRefType input_type, const Type result_elem_type,
      function_ref<InFlightDiagnostic()> emit_error) {
    const unsigned input_bitwidth = getElementTypeBitwidth(input_type);
    const unsigned result_bitwidth = getTypeBitwidth(result_elem_type);
    if (input_bitwidth == result_bitwidth) {
      return MemRefType(
          MemRefType::Builder(input_type).setElementType(result_elem_type));
    }
    const int64_t rank = input_type.getRank();
    if (rank < 2) {
      return emit_error() << "Cannot bitcast along 2nd minor in 1D memref";
    }
    if (input_type.isDynamicDim(rank - 2)) {
      return emit_error() << "Not implemented: Dynamic 2nd minor dimension";
    }
    if (input_type.getDimSize(rank - 2) * input_bitwidth % result_bitwidth !=
        0) {
      return emit_error() << "Input 2nd minor dimension bits not a multiple of "
                             "result bitwidth";
    }
    SmallVector<int64_t> result_shape(input_type.getShape());
    result_shape[rank - 2] =
        input_type.getDimSize(rank - 2) * input_bitwidth / result_bitwidth;

    if (auto affine_layout = dyn_cast<AffineMapAttr>(input_type.getLayout());
        affine_layout && affine_layout.isIdentity()) {
      // An affine map layout is interpreted as "unset"/"unknown" (non-standard
      // semantics)
      return MemRefType(MemRefType::Builder(input_type)
                            .setShape(result_shape)
                            .setElementType(result_elem_type));
    }
    const auto input_layout = dyn_cast<TiledLayoutAttr>(input_type.getLayout());
    if (input_layout == nullptr) {
      return emit_error() << "Expected a tiled layout for the input memref.";
    }
    ArrayRef<xla::Tile> input_tiles = input_layout.getTiles();
    while (!input_tiles.empty() && llvm::all_of(input_tiles.back().dimensions(),
                                                llvm::equal_to<int64_t>(1))) {
      input_tiles = input_tiles.drop_back(1);
    }
    const int64_t num_input_tiles = input_tiles.size();
    for (int64_t i = 0; i < num_input_tiles - 1; ++i) {
      if (input_tiles[i].dimensions().size() <
          input_tiles[i + 1].dimensions().size()) {
        // NOTE: TiledLayoutAttr verification allows tiles that tile across
        // previous levels, like T(256)(128)(2, 1), but it enforces that
        // previous levels are evenly divided by later levels.
        return emit_error() << "Not implemented: Tile at level " << i + 1
                            << " tiles across previous tiles.";
      }
    }

    const bool has_interleaving_tile =
        !input_tiles.empty() && input_tiles.back().dimensions().size() >= 2 &&
        *(input_tiles.back().dimensions().end() - 1) == 1;
    const int64_t input_factor =
        has_interleaving_tile ? *(input_tiles.back().dimensions().end() - 2)
                              : 1;
    if (input_factor * input_bitwidth % result_bitwidth != 0) {
      return emit_error()
             << "Not implemented: Input 2nd minor interleaving tile "
                "bits not a multiple of result bitwidth";
    }
    const int64_t result_factor =
        input_factor * input_bitwidth / result_bitwidth;
    SmallVector<xla::Tile> result_tiles;
    result_tiles.reserve(input_tiles.size());
    for (const xla::Tile& input_tile : input_tiles) {
      const int64_t tile_rank = input_tile.dimensions().size();
      SmallVector<int64_t> tile;
      if (tile_rank == 0) {
        continue;  // Skip
      }
      if (tile_rank == 1) {
        // Recall we checked tiles are of decreasing rank. This is part of a
        // suffix of 1D tiles. The input factor must be 1 since the last tile
        // is not of the form (..., A, 1).
        // The suffix ...(A)(B)(C) becomes ...(1, A)(1, B)(1, C)
        CHECK_EQ(input_factor, 1);
        CHECK_NE(result_factor, 1);  // Identical bitwidths handled above
        tile.push_back(1);
      }
      tile.append(input_tile.dimensions().begin(),
                  input_tile.dimensions().end());
      CHECK_EQ(*(tile.end() - 2) * result_factor % input_factor, 0);
      *(tile.end() - 2) = *(tile.end() - 2) * result_factor / input_factor;
      // Since tiles are of decreasing rank and higher level tiles divide lower
      // level tiles, we can safely remove all-1s tiles.
      if (!llvm::all_of(tile, llvm::equal_to<int64_t>(1))) {
        result_tiles.emplace_back(tile);
      }
    }
    if (!has_interleaving_tile && result_factor != 1) {
      result_tiles.emplace_back(xla::Tile({result_factor, 1}));
    }
    auto result_layout = TiledLayoutAttr::get(
        input_type.getContext(), result_tiles, input_layout.getTileStrides());
    return MemRefType::get(result_shape, result_elem_type, result_layout,
                           input_type.getMemorySpace());
  }

  template <typename Op>
  static LogicalResult verifyStridedOp(Op op, MemRefType memref_ty,
                                       VectorType vector_ty,
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

  template <typename StridedLoadOpT>
  static LogicalResult verifyStridedLoadOp(StridedLoadOpT op) {
    return verifyStridedOp<StridedLoadOpT>(op, op.getBase().getType(),
                                           op.getType(),
                                           /*min_stride=*/0);
  }

  template <typename StridedStoreOpT>
  static LogicalResult verifyStridedStoreOp(StridedStoreOpT op) {
    return verifyStridedOp<StridedStoreOpT>(op, op.getBase().getType(),
                                            op.getValueToStore().getType(),
                                            /*min_stride=*/1);
  }

  template <typename Op>
  static LogicalResult verifyStoreOp(Op op) {
    MemRefType ref_ty = op.getBase().getType();
    if (ref_ty.getMemorySpace() &&
        !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
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
      if (value_ty.getShape() != op.getMask().getType().getShape()) {
        return op.emitOpError("Expected mask shape to match result shape: (")
               << value_ty.getShape() << "). Got: ("
               << op.getMask().getType().getShape() << ").";
      }
    }
    return success();
  }

  template <typename VectorStoreOpT>
  static LogicalResult verifyVectorStoreOp(VectorStoreOpT op) {
    if (!op.getStrides().empty()) {
      return op.emitError(
          "Not implemented: general vector store with strides.");
    }
    MemRefType ref_ty = op.getBase().getType();
    if (llvm::size(op.getIndices()) != ref_ty.getRank()) {
      return op.emitOpError("Expected ") << ref_ty.getRank() << " indices.";
    }
    return verifyStoreOp(op);
  }

  template <typename CompressStoreVregOpT>
  static LogicalResult verifyCompressStoreVregOp(CompressStoreVregOpT op) {
    MemRefType ref_ty = op.getBase().getType();
    if (ref_ty.getMemorySpace() &&
        !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
      return op.emitOpError("Expected base memref to be in VMEM.");
    }
    VectorType value_ty = op.getValueToStore().getType();
    if (value_ty.getElementType() != ref_ty.getElementType()) {
      return op.emitOpError(
          "Expected base and valueToStore element type to match");
    }
    if (llvm::size(op.getIndices()) != ref_ty.getRank()) {
      return op.emitOpError("Expected ") << ref_ty.getRank() << " indices.";
    }
    VectorType mask_ty = op.getMask().getType();
    if (value_ty.getShape()[0] != mask_ty.getShape()[0]) {
      return op.emitOpError(
                 "Expected valueToStore dimension 0 to match mask dimension "
                 "0: ")
             << value_ty.getShape()[0] << " vs " << mask_ty.getShape()[0]
             << ".";
    }
    return success();
  }

  template <typename Op>
  static LogicalResult verifyCompressStoreExpandLoad(
      Op op, VectorType ty, int32_t compress_expand_dim, StringRef vector_name,
      StringRef compress_expand_dim_name) {
    MemRefType ref_ty = op.getBase().getType();
    const int64_t rank = ref_ty.getRank();
    if (ty.getRank() != rank) {
      return op.emitOpError("Expected ")
             << vector_name << " to have the same rank as base (" << rank
             << "). Got: " << ty.getRank() << ".";
    }
    if (llvm::size(op.getIndices()) != rank) {
      return op.emitOpError("Expected ") << rank << " indices.";
    }
    if (compress_expand_dim < 0 || compress_expand_dim >= rank) {
      return op.emitOpError("Expected ")
             << compress_expand_dim_name << " to be in [0, " << rank
             << "). Got: " << compress_expand_dim << ".";
    }
    const StringRef non_target_dim_adjective =
        compress_expand_dim_name == "compress_dim" ? "Non-compressed"
                                                   : "Non-expanded";
    for (int64_t i = 0; i < rank; ++i) {
      if (i == compress_expand_dim || ref_ty.isDynamicDim(i)) {
        continue;
      }
      if (ty.getDimSize(i) > ref_ty.getDimSize(i)) {
        return op.emitOpError(non_target_dim_adjective)
               << " dimension " << i << " of " << vector_name
               << " goes out of bounds of base (" << ty.getDimSize(i) << " > "
               << ref_ty.getDimSize(i) << ").";
      }
    }
    if (ref_ty.getMemorySpace() &&
        !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
      return op.emitOpError("Expected base memref to be in VMEM.");
    }
    if (ty.getElementType() != ref_ty.getElementType()) {
      return op.emitOpError("Expected base and ")
             << vector_name << " element type to match";
    }
    // Note: We deliberately do not go through verifyStoreOp / verifyLoadOp,
    // which reject masked stores/loads of non-32-bit element types. The mask
    // here is mandatory and narrow element types are allowed. They are rejected
    // later, in the vector layout passes.
    if (ty.getShape() != op.getMask().getType().getShape()) {
      return op.emitOpError("Expected mask shape to match ")
             << (vector_name == "valueToStore" ? "value" : "result")
             << " shape: (" << ty.getShape() << "). Got: ("
             << op.getMask().getType().getShape() << ").";
    }
    return success();
  }

  template <typename VectorCompressStoreOpT>
  static LogicalResult verifyVectorCompressStoreOp(VectorCompressStoreOpT op) {
    return verifyCompressStoreExpandLoad(
        op, op.getValueToStore().getType(), op.getCompressDim(),
        /*vector_name=*/"valueToStore",
        /*compress_expand_dim_name=*/"compress_dim");
  }

  template <typename VectorExpandLoadOpT>
  static LogicalResult verifyVectorExpandLoadOp(VectorExpandLoadOpT op) {
    return verifyCompressStoreExpandLoad(
        op, op.getResult().getType(), op.getExpandDim(),
        /*vector_name=*/"result",
        /*compress_expand_dim_name=*/"expand_dim");
  }

  template <typename Op>
  static LogicalResult verifyLoadOp(Op op) {
    MemRefType ref_ty = op.getBase().getType();
    if (ref_ty.getMemorySpace() &&
        !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
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

  template <typename ExpandLoadVregOpT>
  static LogicalResult verifyExpandLoadVregOp(ExpandLoadVregOpT op) {
    MemRefType ref_ty = op.getBase().getType();
    if (ref_ty.getMemorySpace() &&
        !HasMemorySpace(ref_ty, MemorySpace::kVmem)) {
      return op.emitOpError("Expected base memref to be in VMEM.");
    }
    VectorType result_ty = op.getResult().getType();
    if (result_ty.getElementType() != ref_ty.getElementType()) {
      return op.emitOpError("Expected base and result element type to match");
    }
    if (llvm::size(op.getIndices()) != ref_ty.getRank()) {
      return op.emitOpError("Expected ") << ref_ty.getRank() << " indices.";
    }
    // Note: We deliberately do not go through verifyLoadOp, which rejects
    // masked loads of non-32-bit element types. The mask here is mandatory and
    // narrow element types are allowed. They are rejected later, in the vector
    // layout passes.
    VectorType mask_ty = op.getMask().getType();
    if (result_ty.getShape()[0] != mask_ty.getShape()[0]) {
      return op.emitOpError(
                 "Expected result dimension 0 to match mask dimension 0: ")
             << result_ty.getShape()[0] << " vs " << mask_ty.getShape()[0]
             << ".";
    }
    return success();
  }

  template <typename VectorLoadOpT>
  static LogicalResult verifyVectorLoadOp(VectorLoadOpT op) {
    const MemRefType ref_ty = op.getBase().getType();
    if (llvm::size(op.getIndices()) != ref_ty.getRank()) {
      return op.emitOpError("Expected ") << ref_ty.getRank() << " indices.";
    }
    if (!op.getStrides().empty()) {
      if (llvm::size(op.getStrides()) != ref_ty.getRank()) {
        return op.emitOpError("Expected ") << ref_ty.getRank() << " strides.";
      }
      return op.emitError("Not implemented: general vector load with strides.");
    }
    return verifyLoadOp(op);
  }

  template <typename VectorLoadIdxOpT>
  static LogicalResult verifyVectorLoadIdxOp(VectorLoadIdxOpT op) {
    VectorType value_ty = op.getResult().getType();
    MemRefType ref_ty = op.getBase().getType();
    if (llvm::size(op.getIndices()) != ref_ty.getRank()) {
      return op.emitOpError(
                 "Expected one index vector for each dimension of the base "
                 "memref with dimension: ")
             << ref_ty.getRank() << ". Got: " << llvm::size(op.getIndices())
             << ".";
    }
    for (const auto [i, index] : llvm::enumerate(op.getIndices())) {
      VectorType index_ty = llvm::cast<VectorType>(index.getType());
      if (index_ty.getShape() != value_ty.getShape()) {
        return op.emitOpError("Expected ")
               << value_ty.getShape() << " elements in indices. Got "
               << index_ty.getShape() << " in index #" << i << ".";
      }
    }
    return verifyLoadOp(op);
  }

  template <typename VectorStoreIdxOpT>
  static LogicalResult verifyVectorStoreIdxOp(VectorStoreIdxOpT op) {
    VectorType value_ty = op.getValueToStore().getType();
    MemRefType ref_ty = op.getBase().getType();
    if (llvm::size(op.getIndices()) != ref_ty.getRank()) {
      return op.emitOpError(
                 "Expected one index vector for each dimension of the base "
                 "memref with dimension: ")
             << ref_ty.getRank() << ". Got: " << llvm::size(op.getIndices())
             << ".";
    }
    for (const auto [i, index] : llvm::enumerate(op.getIndices())) {
      VectorType index_ty = llvm::cast<VectorType>(index.getType());
      if (index_ty.getShape() != value_ty.getShape()) {
        return op.emitOpError("Expected ")
               << value_ty.getShape() << " elements in indices. Got "
               << index_ty.getShape() << " in index #" << i << ".";
      }
    }
    return verifyStoreOp(op);
  }

  template <typename ReinterpretCastOpT>
  static LogicalResult verifyReinterpretCastOp(ReinterpretCastOpT op) {
    auto source_type = getMemRefType(op.getInput());
    auto target_type = op.getType();
    if (source_type.getMemorySpace() != target_type.getMemorySpace()) {
      return op.emitOpError(
                 "Source and target memory spaces must match, but got ")
             << source_type.getMemorySpace() << " and "
             << target_type.getMemorySpace();
    }
    int64_t num_dynamic_dims = target_type.getNumDynamicDims();
    if (static_cast<int64_t>(op.getDynamicSizes().size()) != num_dynamic_dims) {
      return op.emitOpError("expected ")
             << num_dynamic_dims
             << " dynamic size(s) for the result type, but got "
             << op.getDynamicSizes().size();
    }
    if (auto layout = dyn_cast<TiledLayoutAttr>(target_type.getLayout())) {
      int64_t num_dynamic_strides = layout.getNumDynamicStrides();
      if (op.getDynamicStrides().size() != num_dynamic_strides) {
        return op.emitOpError("expected ")
               << num_dynamic_strides
               << " dynamic stride(s) for the result type, but got "
               << op.getDynamicStrides().size();
      }
    }
    return success();
  }

  template <typename EraseLayoutOpT>
  static LogicalResult verifyEraseLayoutOp(EraseLayoutOpT op) {
    MemRefType operand_type = op.getOperand().getType();
    MemRefType result_type = op.getType();
    if (operand_type.getMemorySpace() != result_type.getMemorySpace()) {
      return op.emitOpError("Cannot change the memref memory space");
    }
    if (auto affine_map_attr = dyn_cast<AffineMapAttr>(result_type.getLayout());
        affine_map_attr == nullptr || !affine_map_attr.isIdentity()) {
      return op.emitOpError("Memref layout must be erased");
    }
    return success();
  }

  template <typename AnnotateOpT>
  static LogicalResult verifyAnnotateOp(AnnotateOpT op) {
    if (op.getNoStore() + op.getNoHazard() + op.getNoHazardNoDeps() > 1) {
      return op.emitOpError(
          "At most one of no_store, no_hazard, or no_hazard_no_deps can be "
          "set");
    }
    auto memref_ty = cast<MemRefType>(op.getOperand().getType());
    if ((op.getNoHazard() || op.getNoHazardNoDeps()) &&
        !HasMemorySpace(memref_ty, MemorySpace::kVmem)) {
      return op.emitOpError(
          "Hazard overrides are only valid for VMEM allocations");
    }
    return success();
  }

  template <typename DynamicRotateOpT>
  static LogicalResult verifyDynamicRotateOp(DynamicRotateOpT op) {
    auto vty = op.getResult().getType();
    if (vty.getRank() <= op.getDimension() || op.getDimension() < 0) {
      return op.emitOpError("Invalid dimension: ") << op.getDimension();
    }
    if (op.getStride().has_value() && op.getStride().value() < 0) {
      return op.emitOpError("Rotate stride must be >= 0 if it is specified");
    }
    if (op.getStrideDimension().has_value() &&
        (vty.getRank() <= op.getStrideDimension().value() ||
         op.getStrideDimension().value() < 0)) {
      return op.emitOpError("Invalid stride dimension: ")
             << op.getStrideDimension().value();
    }
    if (op.getStride().has_value() != op.getStrideDimension().has_value()) {
      return op.emitOpError(
          "Expected either none or both stride and stride dimension are "
          "present");
    }
    return success();
  }

  template <typename ScanCountOpAdaptorT>
  static LogicalResult inferReturnTypesScanCountOp(
      MLIRContext* context, std::optional<Location> location,
      ScanCountOpAdaptorT adaptor,
      ::llvm::SmallVectorImpl<Type>& inferredReturnTypes) {
    inferredReturnTypes.push_back(adaptor.getInMask().getType());
    inferredReturnTypes.push_back(VectorType::get(
        cast<VectorType>(adaptor.getValues().getType()).getShape(),
        IntegerType::get(context, 32)));
    return success();
  }

  template <typename IotaOpT>
  static LogicalResult verifyIotaOp(IotaOpT op) {
    const int64_t rank = op.getType().getRank();
    SmallVector<bool> seen(rank, false);
    for (const int32_t dim : op.getDimensions()) {
      if (dim < 0 || dim >= op.getType().getRank()) {
        return op.emitOpError("Invalid dimension: ") << dim;
      }
      if (seen[dim]) {
        return op.emitOpError("Dimensions must be unique");
      }
      seen[dim] = true;
    }
    return success();
  }

  template <typename MatmulOpT>
  static LogicalResult verifyMatmulOp(MatmulOpT op) {
    // Note - this is not yet an exhaustive verification of matmul. Many of the
    // invariants are spread across infer, apply, llo and below. This is,
    // however, a good start and the recommended place to add more invariants.
    const VectorType lhs_ty = op.getLhs().getType();
    const VectorType rhs_ty = op.getRhs().getType();
    const VectorType acc_ty = op.getAcc().getType();
    const VectorType res_ty = op.getResult().getType();
    if (acc_ty != res_ty) {
      return op.emitOpError(
          "Not implemented: matmul acc and result have different types");
    }
    if (getElementTypeBitwidth(acc_ty) != 32) {
      return op.emitOpError("Expected matmul acc to be 32-bit");
    }

    if (op.getTransposeLhs()) {
      op.emitOpError(
          "Lhs transpose not supported via this API - please use the "
          "dimension numbers API.");
      return failure();
    }

    if (op.getDimensionNumbers().has_value()) {
      auto dimension_numbers = op.getDimensionNumbers().value();
      auto lhs_contracting_dims = dimension_numbers.getLhsContractingDims();
      auto rhs_contracting_dims = dimension_numbers.getRhsContractingDims();
      if (lhs_contracting_dims.size() != rhs_contracting_dims.size() ||
          lhs_contracting_dims.empty()) {
        op.emitOpError(
            "Not implemented: number of lhs and rhs contracting dims must be "
            "equal and non-zero");
        return failure();
      }

      auto lhs_batch_dims = dimension_numbers.getLhsBatchDims();
      auto rhs_batch_dims = dimension_numbers.getRhsBatchDims();

      auto lhs_non_contracting_dims =
          dimension_numbers.getLhsNonContractingDims();
      auto rhs_non_contracting_dims =
          dimension_numbers.getRhsNonContractingDims();

      if (!llvm::is_sorted(lhs_non_contracting_dims)) {
        op.emitOpError(
            "Not implemented: lhs non contracting dims must be sorted");
        return failure();
      }
      if (!llvm::is_sorted(rhs_non_contracting_dims)) {
        op.emitOpError(
            "Not implemented: rhs non contracting dims must be sorted");
        return failure();
      }

      if (lhs_contracting_dims.size() + lhs_non_contracting_dims.size() +
              lhs_batch_dims.size() !=
          lhs_ty.getShape().size()) {
        op.emitOpError(
            "Not implemented: lhs contracting + non contracting + batch dims "
            "must be of the same size as the lhs shape");
        return failure();
      }
      if (rhs_contracting_dims.size() + rhs_non_contracting_dims.size() +
              rhs_batch_dims.size() !=
          rhs_ty.getShape().size()) {
        op.emitOpError(
            "Not implemented: rhs contracting + non contracting + batch dims "
            "must be of the same size as the rhs shape");
        return failure();
      }

      for (int64_t i = 0; i < lhs_contracting_dims.size(); ++i) {
        if (lhs_ty.getDimSize(lhs_contracting_dims[i]) !=
            rhs_ty.getDimSize(rhs_contracting_dims[i])) {
          op.emitOpError(
              "Not implemented: lhs and rhs contracting dims must be of the "
              "same size");
          return failure();
        }
      }

      if (lhs_batch_dims.size() != rhs_batch_dims.size()) {
        op.emitOpError(
            "Not implemented: lhs and rhs should have the same number of batch "
            "dims");
        return failure();
      }

      int64_t lhs_rank = lhs_ty.getShape().size();
      int64_t rhs_rank = rhs_ty.getShape().size();

      std::vector<bool> seen_dims_lhs(lhs_rank, false);
      std::vector<bool> seen_dims_rhs(rhs_rank, false);

      auto check_and_mark_dims = [&](ArrayRef<int64_t> dims,
                                     std::vector<bool>& seen_dims,
                                     StringRef operand) {
        for (int64_t dim : dims) {
          if (dim < 0 || dim >= seen_dims.size()) {
            op.emitOpError("Illegal: Dim ")
                << dim << " is out of bounds for " << operand;
            return failure();
          }
          if (seen_dims[dim]) {
            op.emitOpError("Illegal: Dim ")
                << dim << " repeats in dimension numbers of " << operand;
            return failure();
          }
          seen_dims[dim] = true;
        }
        return success();
      };

      if (failed(check_and_mark_dims(lhs_contracting_dims, seen_dims_lhs,
                                     "lhs")) ||
          failed(check_and_mark_dims(lhs_non_contracting_dims, seen_dims_lhs,
                                     "lhs")) ||
          failed(check_and_mark_dims(lhs_batch_dims, seen_dims_lhs, "lhs"))) {
        return failure();
      }

      if (failed(check_and_mark_dims(rhs_contracting_dims, seen_dims_rhs,
                                     "rhs")) ||
          failed(check_and_mark_dims(rhs_non_contracting_dims, seen_dims_rhs,
                                     "rhs")) ||
          failed(check_and_mark_dims(rhs_batch_dims, seen_dims_rhs, "rhs"))) {
        return failure();
      }

      for (int64_t dim = 0; dim < lhs_rank; ++dim) {
        if (!seen_dims_lhs[dim]) {
          op.emitOpError("Illegal: Dim ")
              << dim << " is not seen in lhs dimension numbers";
          return failure();
        }
      }
      for (int64_t dim = 0; dim < rhs_rank; ++dim) {
        if (!seen_dims_rhs[dim]) {
          op.emitOpError("Illegal: Dim ")
              << dim << " is not seen in rhs dimension numbers";
        }
      }

      for (int64_t i = 0; i < lhs_batch_dims.size(); ++i) {
        int64_t batch_size_lhs = lhs_ty.getDimSize(lhs_batch_dims[i]);
        int64_t batch_size_rhs = rhs_ty.getDimSize(rhs_batch_dims[i]);
        if (batch_size_lhs != batch_size_rhs) {
          op.emitOpError("Not Implemented: batch size must be equal");
          return failure();
        }
        if (batch_size_lhs == 0) {
          op.emitOpError("Illegal: batch size must be > 0");
          return failure();
        }
      }
      auto output_dim_order = dimension_numbers.getOutputDimOrder();
      if (output_dim_order.size() % 2 != 0) {
        op.emitOpError(
            "Illegal: output dim order must have an even number of elements.");
        return failure();
      }

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
        op.emitOpError(
            "Illegal: output dim order must be in the form of [0, "
            "lhs_batch_dims, 0, lhs_non_contracting_dims, 1, "
            "rhs_non_contracting_dims]");
        return failure();
      }
    }
    return success();
  }

  template <typename ConvOpT>
  static LogicalResult verifyConvOp(ConvOpT op) {
    const VectorType lhs_ty = op.getLhs().getType();
    const VectorType rhs_ty = op.getRhs().getType();
    const VectorType acc_ty = op.getAcc().getType();
    const VectorType res_ty = op.getResult().getType();
    if (acc_ty != res_ty) {
      return op.emitOpError("Conv acc and result have different types: ")
             << acc_ty << " vs " << res_ty;
    }

    const auto dnums = op.getDimensionNumbers();
    const int64_t num_spatial = dnums.getInputSpatialDimensions().size();
    const int64_t expected_rank = 2 + num_spatial;

    if (lhs_ty.getRank() != expected_rank) {
      return op.emitOpError(absl::StrFormat(
          "Expected lhs rank to be %d (2 + num_spatial), got %d", expected_rank,
          lhs_ty.getRank()));
    }
    if (rhs_ty.getRank() != expected_rank) {
      return op.emitOpError(absl::StrFormat(
          "Expected rhs rank to be %d (2 + num_spatial), got %d", expected_rank,
          rhs_ty.getRank()));
    }
    if (acc_ty.getRank() != expected_rank) {
      return op.emitOpError(absl::StrFormat(
          "Expected acc rank to be %d (2 + num_spatial), got %d", expected_rank,
          acc_ty.getRank()));
    }

    if (dnums.getKernelSpatialDimensions().size() != num_spatial ||
        dnums.getOutputSpatialDimensions().size() != num_spatial) {
      return op.emitOpError(
          "Expected spatial dimensions count to match across operands");
    }

    auto is_valid_permutation = [](int64_t rank, int64_t d0, int64_t d1,
                                   ArrayRef<int64_t> spatial) {
      llvm::SmallDenseSet<int64_t> dims;
      if (d0 < 0 || d0 >= rank || !dims.insert(d0).second) {
        return false;
      }
      if (d1 < 0 || d1 >= rank || !dims.insert(d1).second) {
        return false;
      }
      for (int64_t d : spatial) {
        if (d < 0 || d >= rank || !dims.insert(d).second) {
          return false;
        }
      }
      return dims.size() == static_cast<size_t>(rank);
    };

    if (!is_valid_permutation(expected_rank, dnums.getInputBatchDimension(),
                              dnums.getInputFeatureDimension(),
                              dnums.getInputSpatialDimensions())) {
      return op.emitOpError("Invalid dimension permutation for lhs");
    }
    if (!is_valid_permutation(expected_rank,
                              dnums.getKernelInputFeatureDimension(),
                              dnums.getKernelOutputFeatureDimension(),
                              dnums.getKernelSpatialDimensions())) {
      return op.emitOpError("Invalid dimension permutation for rhs");
    }
    if (!is_valid_permutation(expected_rank, dnums.getOutputBatchDimension(),
                              dnums.getOutputFeatureDimension(),
                              dnums.getOutputSpatialDimensions())) {
      return op.emitOpError("Invalid dimension permutation for acc/result");
    }

    if (op.getWindowStrides().size() != num_spatial ||
        op.getPadding().size() != 2 * num_spatial ||
        op.getLhsDilation().size() != num_spatial ||
        op.getRhsDilation().size() != num_spatial ||
        op.getWindowReversal().size() != num_spatial) {
      return op.emitOpError(
          "Expected window attributes size to match spatial dims");
    }

    const int64_t feature_group_count = op.getFeatureGroupCount();
    const int64_t batch_group_count = op.getBatchGroupCount();
    if (feature_group_count <= 0) {
      return op.emitOpError("Expected feature_group_count to be positive");
    }
    if (batch_group_count <= 0) {
      return op.emitOpError("Expected batch_group_count to be positive");
    }
    if (feature_group_count > 1 && batch_group_count > 1) {
      return op.emitOpError(
          "At most one of batch_group_count and feature_group_count may be > "
          "1");
    }

    // Contracting feature dimension size match
    const int64_t in_feat = lhs_ty.getDimSize(dnums.getInputFeatureDimension());
    const int64_t kernel_in_feat =
        rhs_ty.getDimSize(dnums.getKernelInputFeatureDimension());
    if (in_feat % feature_group_count != 0) {
      return op.emitOpError(absl::StrFormat(
          "LHS feature dimension size (%d) must be divisible by "
          "feature_group_count (%d)",
          in_feat, feature_group_count));
    }
    if (in_feat / feature_group_count != kernel_in_feat) {
      return op.emitOpError(absl::StrFormat(
          "LHS feature dimension size divided by feature_group_count "
          "(%d / %d = %d) must match kernel input feature dimension size (%d)",
          in_feat, feature_group_count, in_feat / feature_group_count,
          kernel_in_feat));
    }

    // Output feature dimension size match
    const int64_t out_feat =
        acc_ty.getDimSize(dnums.getOutputFeatureDimension());
    const int64_t kernel_out_feat =
        rhs_ty.getDimSize(dnums.getKernelOutputFeatureDimension());
    if (kernel_out_feat % (feature_group_count * batch_group_count) != 0) {
      return op.emitOpError(absl::StrFormat(
          "Kernel output feature dimension size (%d) must be divisible by "
          "feature_group_count * batch_group_count (%d)",
          kernel_out_feat, feature_group_count * batch_group_count));
    }
    if (out_feat != kernel_out_feat) {
      return op.emitOpError(absl::StrFormat(
          "ACC output feature dimension size (%d) must match kernel output "
          "feature dimension size (%d)",
          out_feat, kernel_out_feat));
    }

    // Batch dimension size match
    const int64_t in_batch = lhs_ty.getDimSize(dnums.getInputBatchDimension());
    const int64_t out_batch =
        acc_ty.getDimSize(dnums.getOutputBatchDimension());
    if (in_batch % batch_group_count != 0) {
      return op.emitOpError(
          absl::StrFormat("LHS batch dimension size (%d) must be divisible by "
                          "batch_group_count (%d)",
                          in_batch, batch_group_count));
    }
    if (in_batch / batch_group_count != out_batch) {
      return op.emitOpError(
          absl::StrFormat("LHS batch dimension size divided by "
                          "batch_group_count (%d / %d = %d) "
                          "must match ACC output batch dimension size (%d)",
                          in_batch, batch_group_count,
                          in_batch / batch_group_count, out_batch));
    }

    // Spatial dimension output size formula matching
    for (int64_t i = 0; i < num_spatial; ++i) {
      const int64_t in_dim = dnums.getInputSpatialDimensions()[i];
      const int64_t kernel_dim = dnums.getKernelSpatialDimensions()[i];
      const int64_t out_dim = dnums.getOutputSpatialDimensions()[i];
      const int64_t in_size = lhs_ty.getDimSize(in_dim);
      const int64_t kernel_size = rhs_ty.getDimSize(kernel_dim);
      const int64_t stride = op.getWindowStrides()[i];
      const int64_t pad_low = op.getPadding()[2 * i];
      const int64_t pad_high = op.getPadding()[2 * i + 1];
      const int64_t lhs_dil = op.getLhsDilation()[i];
      const int64_t rhs_dil = op.getRhsDilation()[i];

      if (stride <= 0) {
        return op.emitOpError("Expected window strides to be positive");
      }
      if (lhs_dil <= 0 || rhs_dil <= 0) {
        return op.emitOpError("Expected dilations to be positive");
      }
      const int64_t dilated_input =
          in_size == 0 ? 0 : (in_size - 1) * lhs_dil + 1;
      const int64_t dilated_kernel =
          kernel_size == 0 ? 0 : (kernel_size - 1) * rhs_dil + 1;
      const int64_t padded_input = dilated_input + pad_low + pad_high;
      if (padded_input < dilated_kernel) {
        return op.emitOpError(
            absl::StrFormat("Padded input spatial size (%d) must be at least "
                            "dilated kernel size "
                            "(%d) for spatial dimension %d",
                            padded_input, dilated_kernel, i));
      }
      const int64_t expected_out_size =
          (padded_input - dilated_kernel) / stride + 1;
      if (acc_ty.getDimSize(out_dim) != expected_out_size) {
        return op.emitOpError(absl::StrFormat(
            "Output spatial dimension %d size mismatch: expected %d, got %d",
            out_dim, expected_out_size, acc_ty.getDimSize(out_dim)));
      }
    }

    return success();
  }

  template <typename MaskCastOpT>
  static LogicalResult verifyMaskCastOp(MaskCastOpT op) {
    const VectorType input_ty = op.getInput().getType();
    const VectorType output_ty = op.getResult().getType();
    const int64_t input_rank = input_ty.getRank();
    const int64_t output_rank = output_ty.getRank();
    const int64_t min_rank = std::min(input_rank, output_rank);
    const int64_t max_rank = std::max(input_rank, output_rank);
    CHECK_GE(max_rank, 1);
    // Unfortunately, if input and output ranks are equal, we don't have enough
    // information to determine whether the last dimension is a subelement
    // dimension (packed mask) or a target shape dimension (unpacked mask).
    if (min_rank < max_rank - 1) {
      return op.emitOpError("Input and output ranks must differ by at most 1");
    }
    if (input_ty.getShape().take_front(max_rank - 1) !=
        output_ty.getShape().take_front(max_rank - 1)) {
      return op.emitOpError(
          "Input and output shapes must match on leading dimensions");
    }
    return success();
  }

  template <typename ScanOpT>
  static LogicalResult verifyScanOp(ScanOpT op) {
    VectorType input_ty = op.getInput().getType();
    VectorType output_ty = op.getOutput().getType();

    const int64_t dimension = op.getDimension();
    if (dimension < 0 || dimension >= input_ty.getRank()) {
      return op.emitOpError("Dimension must be in [0, rank).");
    }

    if (input_ty.getElementType().isInteger(1)) {
      if (!output_ty.getElementType().isInteger(32)) {
        return op.emitOpError(
            "Output element type must be i32 vector for i1 vector inputs.");
      }
    } else {
      if (input_ty.getElementType() != output_ty.getElementType()) {
        return op.emitOpError("Input and output element type mismatch.");
      }
    }

    if (input_ty.getShape() != output_ty.getShape()) {
      return op.emitOpError("Input and output shape mismatch. Input shape: (")
             << input_ty.getShape() << "). Output shape: ("
             << output_ty.getShape() << ").";
    }

    if (input_ty.getElementType().isInteger(1) &&
        op.getKind() != ReductionKind::kSum) {
      return op.emitOpError(
          "Only sum reduction is supported for i1 vector inputs.");
    }
    switch (op.getKind()) {
      case ReductionKind::kSum:
        break;
      case ReductionKind::kMaxF:
      case ReductionKind::kMinF:
        if (!isa<FloatType>(input_ty.getElementType())) {
          return op.emitOpError(
              "maxf and minf reductions require float element type.");
        }
        break;
      case ReductionKind::kMaxSI:
      case ReductionKind::kMinSI:
      case ReductionKind::kMaxUI:
      case ReductionKind::kMinUI:
        if (!isa<IntegerType>(input_ty.getElementType())) {
          return op.emitOpError(
              "maxsi, minsi, maxui and minui reductions require integer "
              "element type.");
        }
        break;
      case ReductionKind::kMax_DEPRECATED:
      case ReductionKind::kMin_DEPRECATED:
        return op.emitOpError(
            "max and min reduction kind symbols are deprecated. Max and min "
            "reductions are supported via maxf, minf, maxsi, minsi, maxui, or "
            "minui instead.");
      case ReductionKind::kArgMax:
      case ReductionKind::kArgMin:
      case ReductionKind::kFindFirstSet:
        return op.emitOpError(
            "Only sum, max and min reductions are supported.");
    }

    if (op.getMask() == nullptr) {
      return success();
    }
    if (input_ty.getElementType().isInteger(1)) {
      return op.emitOpError("Mask is not supported for i1 vector inputs.");
    }

    VectorType mask_ty = op.getMask().getType();
    // Enforced via VectorOfRankAndType in .td declaration:
    CHECK_EQ(mask_ty.getRank(), 1);
    if (mask_ty.getDimSize(0) != input_ty.getDimSize(dimension)) {
      return op.emitOpError(
                 "Mask and input mismatch. Expected mask of length: ")
             << input_ty.getDimSize(dimension) << ", but got "
             << mask_ty.getDimSize(0) << ".";
    }

    return success();
  }

  template <typename SortOpT>
  static LogicalResult verifySortOp(SortOpT op) {
    VectorType keys_ty = op.getKeys().getType();
    VectorType values_ty = op.getValues().getType();
    if (keys_ty.getShape() != values_ty.getShape()) {
      return op.emitOpError("Key and value shapes must match: ")
             << keys_ty.getShape() << " vs " << values_ty.getShape();
    }
    if (op.getMask()) {
      VectorType mask_ty = op.getMask().getType();
      if (keys_ty.getShape() != mask_ty.getShape()) {
        return op.emitOpError("Key and input mask shapes must match: ")
               << keys_ty.getShape() << " vs " << mask_ty.getShape();
      }
    }
    VectorType output_mask_ty = op.getOutputMask().getType();
    if (keys_ty.getShape() != output_mask_ty.getShape()) {
      return op.emitOpError("Key and output mask shapes must match: ")
             << keys_ty.getShape() << " vs " << output_mask_ty.getShape();
    }
    if (keys_ty != op.getSortedKeys().getType()) {
      return op.emitOpError("Key and sorted_key types must match: ")
             << keys_ty << " vs " << op.getSortedKeys().getType();
    }
    if (values_ty != op.getSortedValues().getType()) {
      return op.emitOpError("Value and sorted_value types must match: ")
             << values_ty << " vs " << op.getSortedValues().getType();
    }
    return success();
  }

  template <typename GetBarrierSemaphoreOpT>
  static LogicalResult verifyGetBarrierSemaphoreOp(GetBarrierSemaphoreOpT op) {
    MemRefType sem_type = op.getResult().getType();
    if (sem_type.getRank() != 0) {
      op.emitOpError("Barrier semaphore reference must be rank 0");
      return failure();
    }
    return success();
  }

  static bool isRemote(Value device_id, Value core_id) {
    return device_id != nullptr || core_id != nullptr;
  }

  template <typename OpTy>
  static bool isRemote(OpTy op) {
    return isRemote(op.getDeviceId(), op.getCoreId());
  }

  template <typename SemaphoreSignalOpT>
  static LogicalResult verifySemaphoreSignalOp(SemaphoreSignalOpT op) {
    MemRefType sem_type = op.getSemaphore().getType();
    if (sem_type.getRank() != 0) {
      return op.emitOpError("Semaphore reference must be rank 0");
    }
    return success();
  }

  template <typename SemaphoreWaitOpT>
  static LogicalResult verifySemaphoreWaitOp(SemaphoreWaitOpT op) {
    MemRefType sem_type = op.getSemaphore().getType();
    if (sem_type.getRank() != 0) {
      return op.emitOpError("Semaphore reference must be rank 0");
    }
    return success();
  }

  static LogicalResult verifyCommonDmaParams(
      Operation* op, MemRefType source_ty, MemRefType target_ty,
      TypedValue<MemRefType> source_sem, TypedValue<MemRefType> target_sem,
      Value device_id, Value core_id, bool require_memory_spaces,
      int priority) {
    Type sem_elem_type = nullptr;

    if (target_sem) {
      MemRefType target_sem_type = getMemRefType(target_sem);
      if (target_sem_type.getRank() != 0) {
        return op->emitOpError("DMA target semaphore must be rank 0");
      }
      sem_elem_type = target_sem_type.getElementType();
    }

    if (source_sem) {
      MemRefType source_sem_type = getMemRefType(source_sem);
      if (source_sem_type.getRank() != 0) {
        return op->emitOpError("DMA source semaphore reference must be rank 0");
      }
      if (sem_elem_type && source_sem_type.getElementType() != sem_elem_type) {
        return op->emitOpError(
            "DMA source and target semaphore must have the same type");
      }
    }

    if (source_ty.getElementType() != target_ty.getElementType()) {
      return op->emitOpError("DMA source and target element type mismatch");
    }
    if (source_ty.getShape() != target_ty.getShape()) {
      return op->emitOpError("DMA source and target shape mismatch.");
    }

    if (require_memory_spaces) {
      if (source_ty.getMemorySpace() == nullptr) {
        return op->emitOpError(
            "Source memory space must be provided for SparseCore DMA");
      }
      if (target_ty.getMemorySpace() == nullptr) {
        return op->emitOpError(
            "Target memory space must be provided for SparseCore DMA");
      }
    }

    if (priority < 0 || priority > 1) {
      return op->emitOpError(
                 "Not implemented: only support priority 0 or 1, but got ")
             << priority;
    }
    if (priority != 0 && isRemote(device_id, core_id)) {
      return op->emitOpError(
          "Not implemented: non-zero priority is not supported for remote DMA");
    }
    return success();
  }

  template <typename EnqueueDMAOpT>
  static LogicalResult verifyEnqueueDMAOp(EnqueueDMAOpT op,
                                          bool require_memory_spaces) {
    MemRefType source_ty = getMemRefType(op.getSource());
    MemRefType target_ty = getMemRefType(op.getTarget());
    return verifyCommonDmaParams(
        op.getOperation(), source_ty, target_ty, op.getSourceSemaphore(),
        op.getTargetSemaphore(), op.getDeviceId(), op.getCoreId(),
        require_memory_spaces, op.getPriority());
  }

  static FailureOr<bool> isIndirectDMAGather(Operation& op, const Value src,
                                             const Value dst) {
    const auto source_ms = dyn_cast_if_present<MemorySpaceAttr>(
        getMemRefType(src).getMemorySpace());
    const auto target_ms = dyn_cast_if_present<MemorySpaceAttr>(
        getMemRefType(dst).getMemorySpace());
    if (source_ms == nullptr || target_ms == nullptr) {
      return op.emitOpError("Source and target memory spaces must be provided");
    }
    return impl::isGather(op, source_ms.getValue(), source_ms.getCoreType(),
                          target_ms.getValue(), target_ms.getCoreType());
  }

  static FailureOr<bool> isGather(EnqueueIndirectDMAOp op) {
    return isIndirectDMAGather(*op.getOperation(), op.getSource(),
                               op.getTarget());
  }

  template <typename EnqueueIndirectDMAOpT>
  static LogicalResult verifyEnqueueIndirectDMAOp(EnqueueIndirectDMAOpT op) {
    const MemRefType source_ty = getMemRefType(op.getSource());
    const MemRefType target_ty = getMemRefType(op.getTarget());

    if (source_ty.getElementType() != target_ty.getElementType()) {
      return op.emitOpError("Source and target element type mismatch");
    }

    FAILUREOR_ASSIGN_OR_RETURN(bool is_gather, isGather(op));

    const Value offsets = op.getOffsets();
    ArrayRef<int64_t> offsets_shape;
    if (auto offsets_ty = dyn_cast<MemRefType>(offsets.getType());
        offsets_ty != nullptr) {
      if (!HasMemorySpace(offsets_ty, MemorySpace::kVmem)) {
        return op.emitOpError("Offsets memref must be in VMEM");
      }
      offsets_shape = offsets_ty.getShape();
    } else if (auto offsets_ty = dyn_cast<VectorType>(offsets.getType());
               offsets_ty != nullptr) {
      // Vector offsets may have trailing padding lanes. The local memref's
      // leading dim is the active prefix length, so it must be static and in
      // [1, num_offsets]: only those rows are transferred, and
      // wait_indirect_dma counts bytes from the local memref.
      const MemRefType on_tile_ty = is_gather ? target_ty : source_ty;
      const int64_t num_offsets = offsets_ty.getDimSize(0);
      if (on_tile_ty.getRank() == 0 || on_tile_ty.isDynamicDim(0) ||
          on_tile_ty.getDimSize(0) < 1 ||
          on_tile_ty.getDimSize(0) > num_offsets) {
        return op.emitOpError() << "Leading dimension of the "
                                << (is_gather ? "target (gather result)"
                                              : "source (scatter updates)")
                                << " must be static and in [1, " << num_offsets
                                << "], got " << on_tile_ty;
      }
      offsets_shape = on_tile_ty.getShape().take_front(1);
    } else {
      return op.emitOpError("Offsets must be a memref or vector type");
    }

    if (MemRefType sem_ty = getMemRefType(op.getSemaphore());
        sem_ty.getRank() != 0) {
      return op.emitOpError("Semaphore must be rank 0");
    }

    if (is_gather) {
      return impl::verifyGather(op.getOperation(),
                                /*operand_shape=*/source_ty.getShape(),
                                /*offsets_shape=*/offsets_shape,
                                /*result_shape=*/target_ty.getShape());
    }
    return impl::verifyScatter(op.getOperation(),
                               /*updates_shape=*/source_ty.getShape(),
                               /*offsets_shape=*/offsets_shape,
                               /*operand_shape=*/target_ty.getShape());
  }

  template <typename WaitDMA2OpT>
  static LogicalResult verifyWaitDMA2Op(WaitDMA2OpT op) {
    MemRefType sem_type = op.getSemaphore().getType();
    if (sem_type.getRank() != 0) {
      return op.emitOpError("DMA wait semaphore must be rank 0");
    }
    return success();
  }

  template <typename WaitDMAOpT>
  static LogicalResult verifyWaitDMAOp(WaitDMAOpT op,
                                       bool require_memory_spaces) {
    MemRefType source_ty = getMemRefType(op.getSource());
    MemRefType target_ty = getMemRefType(op.getTarget());
    return verifyCommonDmaParams(
        op.getOperation(), source_ty, target_ty, op.getSourceSemaphore(),
        op.getTargetSemaphore(), op.getDeviceId(), op.getCoreId(),
        require_memory_spaces, op.getPriority());
  }

  static FailureOr<bool> isGather(WaitIndirectDMAOp op) {
    return isIndirectDMAGather(*op.getOperation(), op.getSrc(), op.getDst());
  }

  template <typename WaitIndirectDMAOpT>
  static LogicalResult verifyWaitIndirectDMAOp(WaitIndirectDMAOpT op) {
    MemRefType sem_type = getMemRefType(op.getSemaphore());
    if (sem_type.getRank() != 0) {
      return op.emitOpError("Indirect DMA wait semaphore must be rank 0");
    }
    return isGather(op);
  }

  template <typename RegionOpT>
  static LogicalResult verifyRegionOp(RegionOpT op) {
    for (auto result_type : op.getResultTypes()) {
      if (!isa<FloatType, IntegerType, VectorType, IndexType>(result_type)) {
        return op.emitOpError(
            "Region result must be float, int, index or a vector type.");
      }
    }
    return success();
  }

  template <typename ShuffledLoadOpT>
  static LogicalResult verifyShuffledLoadOp(ShuffledLoadOpT op) {
    if (op.getBase().getType().getRank() != op.getIndices().size()) {
      return op.emitOpError(
                 "Base memref's rank and indices size do not match: ")
             << op.getBase().getType().getRank() << " vs "
             << op.getIndices().size();
    }
    if (op.getSublaneMask().size() != op.getType().getShape()[0]) {
      return op.emitOpError("Expected sublane mask size equals to ")
             << op.getType().getShape()[0] << " but got "
             << op.getSublaneMask().size();
    }
    if (op.getSublaneOffsets().size() != op.getType().getShape()[0]) {
      return op.emitOpError("Expected sublane offsets size equals to ")
             << op.getType().getShape()[0] << " but got "
             << op.getSublaneOffsets().size();
    }
    return success();
  }

  template <typename ShuffledStoreOpT>
  static LogicalResult verifyShuffledStoreOp(ShuffledStoreOpT op) {
    if (op.getBase().getType().getRank() != op.getIndices().size()) {
      return op.emitOpError(
                 "Base memref's rank and indices size do not match: ")
             << op.getBase().getType().getRank() << " vs "
             << op.getIndices().size();
    }
    if (op.getValueToStore().getType().getRank() != op.getIndices().size()) {
      return op.emitOpError(
                 "The rank of value to store and indices size do not match: ")
             << op.getBase().getType().getRank() << " vs "
             << op.getIndices().size();
    }
    if (op.getSublaneMask().size() !=
        op.getValueToStore().getType().getShape()[0]) {
      return op.emitOpError("Expected sublane mask size equals to ")
             << op.getValueToStore().getType().getShape()[0] << " but got "
             << op.getSublaneMask().size();
    }
    if (op.getSublaneOffsets().size() !=
        op.getValueToStore().getType().getShape()[0]) {
      return op.emitOpError("Expected sublane offsets size equals to ")
             << op.getValueToStore().getType().getShape()[0] << " but got "
             << op.getSublaneOffsets().size();
    }
    return success();
  }

  template <typename ConcatenateOpT>
  static LogicalResult verifyConcatenateOp(ConcatenateOpT op) {
    auto dimension = op.getDimension();
    if (op.getOperands().size() < 2) {
      return op.emitOpError("Expected at least 2 operands for concatenate op.");
    }
    auto first_type = cast<VectorType>(op.getOperand(0).getType());
    auto first_shape = first_type.getShape();
    auto first_dtype = first_type.getElementType();
    for (auto operand : op.getOperands()) {
      auto vty = dyn_cast<VectorType>(operand.getType());
      if (!vty) {
        return op.emitOpError("Operand must be a vector type.");
      }
      auto shape = vty.getShape();
      auto dtype = vty.getElementType();
      if (dtype != first_dtype) {
        return op.emitOpError(
            "Not implemented:: Expected all operands to have the same element "
            "type.");
      }
      if (shape.size() != first_shape.size()) {
        return op.emitOpError("Expected all operands to have the same rank.");
      }
      for (int dim = 0; dim < shape.size(); ++dim) {
        if (dim != dimension && shape[dim] != first_shape[dim]) {
          return op.emitOpError(
              "Not implemented: Expected all operands to have "
              "the same shape outside of the concat dim");
        }
      }
    }
    return success();
  }

  template <typename ConcatenateOpAdaptorT>
  static LogicalResult inferReturnTypesConcatenateOp(
      MLIRContext* context, std::optional<Location> location,
      ConcatenateOpAdaptorT adaptor,
      SmallVectorImpl<Type>& inferredReturnTypes) {
    ValueRange operands = adaptor.getSources();
    if (operands.empty()) {
      return failure();
    }
    auto dimension = adaptor.getDimension();
    auto first_type = dyn_cast<VectorType>(operands[0].getType());
    if (!first_type) {
      return failure();
    }
    for (auto [i, operand] : llvm::enumerate(operands)) {
      if (auto operand_ty = dyn_cast<VectorType>(operand.getType());
          !operand_ty || operand_ty.getRank() <= dimension ||
          operand_ty.getRank() != first_type.getRank()) {
        return failure();
      }
    }
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

  template <typename WeirdOpT>
  static LogicalResult verifyWeirdOp(WeirdOpT op) {
    const mlir::Type in_type = op.getInput().getType();
    if (const auto in_vec_type =
            dyn_cast<VectorType>(in_type)) {  // Vector case.
      if (!in_vec_type.getElementType().isF32()) {
        return op.emitOpError("Input type must be F32");
      }
      const mlir::Type out_type = op.getResult().getType();
      const auto out_vec_type = dyn_cast<VectorType>(out_type);
      if (!out_vec_type) {
        return op.emitOpError(
            "Output type must be a vector when input is a vector");
      }
      if (!out_vec_type.getElementType().isInteger(1)) {
        return op.emitOpError("Output type must be I1");
      }
    } else {  // Scalar case.
      if (!in_type.isF32()) {
        return op.emitOpError("Input type must be F32");
      }
      const mlir::Type out_type = op.getResult().getType();
      if (!out_type.isInteger(1)) {
        return op.emitOpError("Output type must be I1 scalar");
      }
    }
    return success();
  }

  template <typename LogBufferOpT>
  static LogicalResult verifyLogBufferOp(LogBufferOpT op) {
    const MemRefType input_type = op.getInput().getType();
    if (input_type.getRank() != op.getShape().size()) {
      return op.emitOpError(
          "Shape must have the same length as the rank of the input");
    }
    return success();
  }

  template <typename ReciprocalOpT>
  static LogicalResult verifyReciprocalOp(ReciprocalOpT op) {
    auto elem_ty = op.getType().getElementType();
    if (!(elem_ty.isF32() || elem_ty.isBF16())) {
      return op.emitOpError(
          "Not implemented: Reciprocal op for non-f32/bf16 dtypes");
    }
    return success();
  }

  template <typename UnpackSubelementsOpT>
  static LogicalResult verifyUnpackSubelementsOp(UnpackSubelementsOpT op) {
    const int packing_factor = getElementTypeBitwidth(op.getType()) /
                               getElementTypeBitwidth(op.getSource().getType());
    if (auto index = op.getIndex(); index >= packing_factor) {
      return op.emitOpError("Index must be between 0 and the packing factor (")
             << packing_factor << "), got " << index;
    }
    if (op.getUnsignedIntegers() &&
        !op.getSource().getType().getElementType().isSignlessInteger()) {
      return op.emitOpError(
          "unsigned_integers can only be set when the source type is an "
          "integer");
    }
    return success();
  }

  template <typename PackSubelementsOpT>
  static LogicalResult verifyPackSubelementsOp(PackSubelementsOpT op) {
    return verifyPackOp(op);
  }

  template <typename PackMaskOpT>
  static LogicalResult verifyPackMaskOp(PackMaskOpT op) {
    return verifyPackOp(op);
  }

  static LogicalResult verifyElementwisePacking(Operation* op, Type unpacked_ty,
                                                Type packed_ty) {
    if (!(unpacked_ty.isF32() && packed_ty.isBF16()) &&
        !(unpacked_ty.isSignlessInteger() && packed_ty.isSignlessInteger())) {
      return op->emitOpError(
          "Only packing/unpacking f32 <-> bf16 and integer <-> integer is "
          "supported");
    }
    return success();
  }

  template <typename PackElementwiseOpT>
  static LogicalResult verifyPackElementwiseOp(PackElementwiseOpT op) {
    if (op.getSources().empty()) {
      return op.emitOpError("At least one source is required");
    }
    const auto src_vty = cast<VectorType>(op.getSources().front().getType());
    if (getElementTypeBitwidth(src_vty) !=
        getElementTypeBitwidth(op.getType())) {
      return op.emitOpError(
          "All sources must have the same bitwidth as the result");
    }
    if (!op.getType().getElementType().isSignlessInteger()) {
      return op.emitOpError("Output type must be a signless integer type");
    }

    auto src_elem_ty = src_vty.getElementType();
    auto tgt_elem_ty = op.getTargetType();
    if (failed(verifyElementwisePacking(op, /*unpacked_ty=*/src_elem_ty,
                                        /*packed_ty=*/tgt_elem_ty))) {
      return failure();
    }
    const int packing_factor =
        getElementTypeBitwidth(src_vty) / getTypeBitwidth(tgt_elem_ty);
    if (packing_factor != op.getSources().size()) {
      return op.emitOpError(
                 "The number of sources must match the packing factor (")
             << packing_factor << "), got " << op.getSources().size();
    }
    return success();
  }

  template <typename UnpackElementwiseOpT>
  static LogicalResult verifyUnpackElementwiseOp(UnpackElementwiseOpT op) {
    if (failed(verifyElementwisePacking(
            op, /*unpacked_ty=*/op.getType().getElementType(),
            /*packed_ty=*/op.getSourceType()))) {
      return failure();
    }
    const int packing_factor = getElementTypeBitwidth(op.getType()) /
                               getTypeBitwidth(op.getSourceType());
    if (auto index = op.getIndex(); index >= packing_factor) {
      return op.emitOpError("Index must be between 0 and the packing factor (")
             << packing_factor << "), got " << index;
    }
    return success();
  }

  template <typename DynamicGatherOpT>
  static LogicalResult verifyDynamicGatherOp(DynamicGatherOpT op) {
    const int64_t rank = op.getSource().getType().getRank();
    SmallVector<bool> seen(rank, false);
    for (int32_t d : op.getDimensions()) {
      if (d < 0 || d >= rank) {
        return op.emitOpError("Dimensions must be in [0, rank), but got ") << d;
      }
      if (seen[d]) {
        return op.emitOpError("Dimensions must be unique");
      }
      seen[d] = true;
    }
    const ArrayRef<int64_t> source_shape = op.getSource().getType().getShape();
    const ArrayRef<int64_t> result_shape = op.getType().getShape();
    if (source_shape.size() != result_shape.size()) {
      return op.emitOpError("Source and result shapes must have the same rank");
    }
    for (int32_t i = 0; i < source_shape.size(); ++i) {
      if (!seen[i] && source_shape[i] != result_shape[i]) {
        return op.emitOpError(
            "Source and result shapes must match on non-gather dimensions");
      }
    }
    return success();
  }

  template <typename DynamicGatherOpAdaptorT>
  static LogicalResult inferReturnTypesDynamicGatherOp(
      MLIRContext* context, std::optional<Location> location,
      DynamicGatherOpAdaptorT adaptor,
      SmallVectorImpl<Type>& inferredReturnTypes) {
    VectorType source_vty = cast<VectorType>(adaptor.getSource().getType());
    VectorType indices_vty = cast<VectorType>(adaptor.getIndices().getType());
    inferredReturnTypes.push_back(
        VectorType::get(indices_vty.getShape(), source_vty.getElementType()));
    return success();
  }

  template <typename AllReduceOpT>
  static LogicalResult verifyAllReduceOp(AllReduceOpT op) {
    auto in_ty = op.getInput().getType();
    auto in_bitwidth = getElementTypeBitwidth(in_ty);
    auto out_ty = op.getOutput().getType();
    auto kind = op.getKind();

    if (in_bitwidth == 1) {
      // For mask vectors, the single (semantically scalar) result is broadcast
      // into a vector of ints of whatever shape the target supports (not
      // necessarily the same as the input).
      if (!out_ty.getElementType().isSignlessInteger()) {
        return op.emitOpError(
            "Vector mask all-reduce must have integer output");
      }
      switch (kind) {
        case ReductionKind::kSum:
        case ReductionKind::kFindFirstSet:
          break;
        default:
          return op.emitOpError(
              "Mask all-reduce only supports sum and find_first_set kinds");
      }
      return success();
    }

    switch (kind) {
      case ReductionKind::kSum:
        if (in_ty != out_ty) {
          return op.emitOpError(
              "Sum, max, and min reductions must have the same "
              "input and output type");
        }
        break;
      case ReductionKind::kMaxF:
      case ReductionKind::kMinF:
        if (!isa<FloatType>(in_ty.getElementType())) {
          return op.emitOpError(
              "maxf and minf reductions require float element type");
        }
        if (in_ty != out_ty) {
          return op.emitOpError(
              "Sum, max, and min reductions must have the same "
              "input and output type");
        }
        break;
      case ReductionKind::kMaxSI:
      case ReductionKind::kMinSI:
      case ReductionKind::kMaxUI:
      case ReductionKind::kMinUI:
        if (!isa<IntegerType>(in_ty.getElementType())) {
          return op.emitOpError(
              "maxsi, minsi, maxui and minui reductions require integer "
              "element type");
        }
        if (in_ty != out_ty) {
          return op.emitOpError(
              "Sum, max, and min reductions must have the same "
              "input and output type");
        }
        break;
      case ReductionKind::kMax_DEPRECATED:
      case ReductionKind::kMin_DEPRECATED:
        return op.emitOpError(
            "max and min reduction kind symbols are deprecated. Max and min "
            "reductions are supported via maxf, minf, maxsi, minsi, maxui, or "
            "minui instead");
      case ReductionKind::kArgMax:
      case ReductionKind::kArgMin:
        if (in_ty.getShape() != out_ty.getShape()) {
          return op.emitOpError(
              "Arg_max and arg_min "
              "must have the same input and output shape");
        }
        if (!in_ty.getElementType().isF32() &&
            !in_ty.getElementType().isBF16()) {
          return op.emitOpError(
              "Not Implemented: Only f32 and bf16 input is supported for "
              "arg_max and arg_min");
        }
        if (!out_ty.getElementType().isSignlessInteger(in_bitwidth)) {
          return op.emitOpError(absl::StrFormat(
              "Arg_max and arg_min must have i%d output", in_bitwidth));
        }
        if (in_ty.getShape()[op.getDim()] > (1LL << in_bitwidth)) {
          return op.emitOpError(absl::StrFormat(
              "Reduction dimension size (%d) exceeds 2^%d (%d), which "
              "overflows %d-bit indices for arg_max/arg_min",
              in_ty.getShape()[op.getDim()], in_bitwidth, 1LL << in_bitwidth,
              in_bitwidth));
        }
        break;
      case ReductionKind::kFindFirstSet:
        return op.emitOpError("Only i1 input is supported for find_first_set");
    }
    return success();
  }

  template <typename ReduceIndexOpT>
  static LogicalResult verifyReduceIndexOp(ReduceIndexOpT op) {
    auto in_ty = op.getInput().getType();
    auto out_ty = op.getOutput().getType();
    auto bitwidth = getElementTypeBitwidth(in_ty);
    auto axis = op.getAxis();
    auto kind = op.getKind();
    const bool is_arg_max_or_min =
        kind == ReductionKind::kArgMax || kind == ReductionKind::kArgMin;
    const bool is_find_first_set = kind == ReductionKind::kFindFirstSet;
    if (!is_arg_max_or_min && !is_find_first_set) {
      return op.emitOpError(
          "Reduction kind must be arg_max, arg_min, or find_first_set");
    }
    if (is_arg_max_or_min && !in_ty.getElementType().isF32() &&
        !in_ty.getElementType().isBF16()) {
      return op.emitOpError(
          "Not Implemented: Only f32 and bf16 input is supported for arg_max "
          "and arg_min");
    }
    if (is_arg_max_or_min &&
        !out_ty.getElementType().isSignlessInteger(bitwidth)) {
      return op.emitOpError(absl::StrFormat(
          "Arg_max and arg_min must have i%d output", bitwidth));
    }
    if (is_find_first_set && (!in_ty.getElementType().isSignlessInteger(1) ||
                              !out_ty.getElementType().isSignlessInteger(32))) {
      return op.emitOpError(
          "find_first_set requires i1 input and i32 output element types");
    }

    // SameOperandsAndResultRank checks the following:
    CHECK_EQ(in_ty.getRank(), out_ty.getRank());

    auto in_shape = in_ty.getShape();
    auto out_shape = out_ty.getShape();
    if (axis < 0 || axis >= in_shape.size()) {
      return op.emitOpError("Axis must be in [0, ")
             << in_shape.size() << "), but got " << axis;
    }
    // The largest possible result is the last index along the reduced dimension
    // for arg_max/arg_min, and the dimension size for find_first_set (returned
    // when no element is set). It must fit in the output element type, treated
    // as unsigned.
    const int64_t max_result =
        is_find_first_set ? in_shape[axis] : in_shape[axis] - 1;
    const unsigned out_bitwidth = out_ty.getElementTypeBitWidth();
    if (static_cast<uint64_t>(max_result) > llvm::maxUIntN(out_bitwidth)) {
      return op.emitOpError("Reduction dimension size ")
             << in_shape[axis] << " is too large for "
             << stringifyReductionKind(kind) << ": the largest result "
             << max_result << " does not fit in " << out_bitwidth
             << "-bit unsigned indices";
    }

    for (int64_t i = 0; i < in_ty.getRank(); ++i) {
      const int64_t expected_output_size = (i == axis) ? 1 : in_shape[i];
      if (out_shape[i] != expected_output_size) {
        return op.emitOpError("Expected output dimension ")
               << i << " to have size " << expected_output_size << ", but got "
               << out_shape[i];
      }
    }
    return success();
  }

  template <typename ReduceOpT>
  static LogicalResult verifyReduceOp(ReduceOpT op) {
    VectorType input_type = op.getInput().getType();
    VectorType output_type = op.getOutput().getType();

    // SameOperandsAndResultRank checks the following:
    CHECK_EQ(input_type.getRank(), output_type.getRank());
    const int64_t rank = input_type.getRank();

    SmallVector<bool> is_reduced(rank, false);
    for (int64_t dim : op.getDimensions()) {
      if (dim < 0 || dim >= rank) {
        return op.emitOpError("Reduced dimension ")
               << dim << " is out of bounds [0, " << rank << ")";
      }
      if (is_reduced[dim]) {
        return op.emitOpError("Reduced dimension ")
               << dim << " is present more than once";
      }
      is_reduced[dim] = true;
    }

    const ArrayRef<int64_t> input_shape = input_type.getShape();
    const ArrayRef<int64_t> output_shape = output_type.getShape();
    for (int64_t i = 0; i < rank; ++i) {
      const int64_t expected_output_size = is_reduced[i] ? 1 : input_shape[i];
      if (output_shape[i] != expected_output_size) {
        return op.emitOpError("Expected output dimension ")
               << i << " to have size " << expected_output_size << ", but got "
               << output_shape[i];
      }
    }

    const Type input_element_type = input_type.getElementType();
    const Type output_element_type = output_type.getElementType();
    switch (op.getKind()) {
      case ReductionKind::kArgMax:
      case ReductionKind::kArgMin:
        return op.emitOpError(
            "arg_max/arg_min not supported - use tpu.reduce_index instead");
      case ReductionKind::kFindFirstSet:
        return op.emitOpError("find_first_set not supported");
      case ReductionKind::kMax_DEPRECATED:
      case ReductionKind::kMin_DEPRECATED:
        return op.emitOpError(
            "max and min reduction kind symbols are deprecated. Max and min "
            "reductions are supported via maxf, minf, maxsi, minsi, maxui, or "
            "minui instead");
      case ReductionKind::kMaxF:
      case ReductionKind::kMinF:
        if (!isa<FloatType>(input_element_type)) {
          return op.emitOpError(
              "maxf and minf reductions require float element type");
        }
        if (input_element_type != output_element_type) {
          return op.emitOpError(
              "Input and output must have the same element type for sum, max "
              "and min reductions");
        }
        break;
      case ReductionKind::kMaxSI:
      case ReductionKind::kMinSI:
      case ReductionKind::kMaxUI:
      case ReductionKind::kMinUI:
        if (!isa<IntegerType>(input_element_type)) {
          return op.emitOpError(
              "maxsi, minsi, maxui and minui reductions require integer "
              "element type");
        }
        if (input_element_type != output_element_type) {
          return op.emitOpError(
              "Input and output must have the same element type for sum, max "
              "and min reductions");
        }
        break;
      case ReductionKind::kSum:
        // TODO(tlongeri): Might be worth allowing things like bf16 -> f32.
        if (input_element_type != output_element_type) {
          return op.emitOpError(
              "Input and output must have the same element type for sum, max "
              "and min reductions");
        }
        break;
    }

    return success();
  }

  template <typename AssumeMultipleOpT>
  static LogicalResult verifyAssumeMultipleOp(AssumeMultipleOpT op) {
    if (op.getMultiple() < 1) {
      return op.emitError("Multiple must be >= 1, got ") << op.getMultiple();
    }
    if (auto value = mlir::getConstantIntValue(op.getValue());
        value.has_value() && (*value % op.getMultiple() != 0)) {
      return op.emitError("Operand is a constant ")
             << *value << " that is not a multiple of " << op.getMultiple();
    }
    return success();
  }

  template <typename SublaneShuffleOpT>
  static LogicalResult verifySublaneShuffleOp(SublaneShuffleOpT op) {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto result = op.getResult();
    auto lhs_ty = dyn_cast<VectorType>(lhs.getType());
    auto rhs_ty = dyn_cast<VectorType>(rhs.getType());
    auto result_ty = dyn_cast<VectorType>(result.getType());

    if (!lhs_ty || !rhs_ty || !result_ty) {
      return op.emitOpError("Expected operands and result to be vector types");
    }

    if (lhs_ty.getShape() != rhs_ty.getShape() ||
        lhs_ty.getShape() != result_ty.getShape()) {
      return op.emitOpError("Expected lhs, rhs, and result shapes to match");
    }
    if (lhs_ty.getElementType() != rhs_ty.getElementType() ||
        lhs_ty.getElementType() != result_ty.getElementType()) {
      return op.emitOpError(
          "Expected lhs, rhs, and result element types to match");
    }

    auto pattern = op.getPattern();
    auto shape = result_ty.getShape();
    if (shape.size() < 2 || shape.size() > 3) {
      return op.emitOpError("Vreg rank should be 2 or 3");
    }
    auto sublane_count = shape[0];

    if (pattern.size() != sublane_count) {
      return op.emitOpError("Expected pattern size (")
             << pattern.size() << ") to match result/operand sublanes ("
             << sublane_count << ")";
    }

    int64_t total_input_sublanes = sublane_count * 2;
    for (int32_t idx : pattern) {
      if (idx < 0 || idx >= total_input_sublanes) {
        return op.emitOpError("Pattern index ")
               << idx << " out of bounds [0, " << (total_input_sublanes - 1)
               << "]";
      }
    }
    return success();
  }

  template <typename ReshapeOpT>
  static LogicalResult verifyReshapeOp(ReshapeOpT op) {
    auto src_ty = op.getSource().getType();
    auto dst_ty = op.getResult().getType();
    if (src_ty.getElementType() != dst_ty.getElementType()) {
      return op.emitOpError("element type must match");
    }
    if (src_ty.getNumElements() != dst_ty.getNumElements()) {
      return op.emitOpError() << "element count must match";
    }
    return success();
  }

  template <typename StochasticConvertElementwiseOpT>
  static LogicalResult verifyStochasticConvertElementwiseOp(
      StochasticConvertElementwiseOpT op) {
    auto dst_ty = op.getDstType();
    if (!llvm::isa<mlir::BFloat16Type, mlir::Float8E5M2Type,
                   mlir::Float8E4M3FNType, mlir::Float8E4M3B11FNUZType,
                   mlir::Float4E2M1FNType>(dst_ty) &&
        !dst_ty.isSignlessInteger(4)) {
      return op.emitOpError("Unsupported destination type: ") << dst_ty;
    }
    return success();
  }

  template <typename FetchAndAddSyncOpT>
  static LogicalResult verifyFetchAndAddSyncOp(FetchAndAddSyncOpT op) {
    MemRefType base_type = op.getBase().getType();

    if (base_type.getRank() != op.getIndices().size()) {
      return op.emitOpError("Number of indices (")
             << op.getIndices().size() << ") must match memref rank ("
             << base_type.getRank() << ")";
    }
    // TODO(slebedev): Require the base to be in SMEM.
    // TODO(slebedev): Check that the enclosing function has subcore_parallel
    // in its dimension semantics.
    return success();
  }

  template <typename TileSizeOpT>
  static LogicalResult verifyTileSizeOp(TileSizeOpT op) {
    MemRefType memref_ty = op.getSource().getType();
    const int32_t index = op.getIndex();
    if (index < 0 || index >= memref_ty.getRank()) {
      return op.emitOpError("Index out of bounds");
    }
    return success();
  }

  template <typename MemRefMemorySpaceIsOpT>
  static LogicalResult verifyMemRefMemorySpaceIsOp(MemRefMemorySpaceIsOpT op) {
    if (op.getMemorySpace().getValue() == MemorySpace::kAny) {
      return op.emitOpError("Comparing to 'any' can never succeed");
    }
    return success();
  }
};

}  // namespace mlir::tpu::impl

#endif  // XLA_MOSAIC_DIALECT_TPU_TPU_OPS_IMPL_H_

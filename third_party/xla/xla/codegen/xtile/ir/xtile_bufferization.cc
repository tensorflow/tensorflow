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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/codegen/emitters/implicit_arith_op_builder.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/mlir/utils/math_util.h"

namespace xla::xtile {

static llvm::SmallVector<mlir::OpFoldResult> GetStaticFoldResult(
    mlir::OpBuilder& builder, llvm::ArrayRef<int64_t> input) {
  return llvm::map_to_vector(input, [&builder](int64_t value) {
    return mlir::OpFoldResult(builder.getIndexAttr(value));
  });
}

// Get the size of the memref subview with the output size clamped to inbound
// elements, if full_size is true then unit values are inserted for reduced
// dimensions.
// The derivation of these bounds is as follows:
//   index + tile_size * stride <= size - 1
//   tile_size * stride <= size - 1 - index
//   tile_size <= size - 1 - index / stride
//   tile_size < ((size - 1 - index) / stride) + 1
static llvm::SmallVector<mlir::OpFoldResult> GetClampedTileSize(
    mlir::ImplicitLocOpBuilder& builder, TiledBufferInterface op,
    bool full_size) {
  llvm::SmallVector<mlir::OpFoldResult> tile_size;
  llvm::SmallDenseSet<unsigned> reduced_dims = op.getReducedDimensions();
  int64_t idx = 0;
  for (auto [buffer_size, offset, stride, full_tile_size] :
       llvm::zip(op.getBuffer().getType().getShape(), op.getOffsets(),
                 op.getStrides(), op.getFullTileShape())) {
    if (reduced_dims.contains(idx++)) {
      if (full_size) {
        tile_size.emplace_back(builder.getIndexAttr(1));
      }
      continue;
    }
    emitters::ImplicitArithOpBuilder arith_builder(
        mlir::arith::ConstantIndexOp::create(builder, (buffer_size - 1)),
        &builder);
    auto numerator = arith_builder - offset;
    // The stride can be 0 for single element tiles.
    // TODO(willfroom): Fix tile analysis so this never happens.
    auto clamped_stride = std::max<int64_t>(stride, 1);
    auto bound = numerator / clamped_stride + 1;
    tile_size.emplace_back(bound.min(full_tile_size));
  }

  return tile_size;
}

// Get the subview of the op buffer with its size clamped such that all elements
// are in bounds.
static mlir::TypedValue<mlir::MemRefType> GetClampedSubView(
    mlir::ImplicitLocOpBuilder& builder, TiledBufferInterface op) {
  auto tile_size = GetClampedTileSize(builder, op, true);

  auto offsets = mlir::getAsOpFoldResult(op.getOffsets());
  auto strides = GetStaticFoldResult(builder, op.getStrides());

  mlir::RankedTensorType tile_type = op.getTile().getType();
  llvm::SmallVector<int64_t> output_shape(tile_type.getRank(),
                                          mlir::ShapedType::kDynamic);
  mlir::MemRefType subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          output_shape, op.getBuffer().getType(), offsets, tile_size, strides);

  return mlir::memref::SubViewOp::create(builder, subview_type, op.getBuffer(),
                                         offsets, tile_size, strides);
}

// Gets the subview of the op buffer with the precondition that the tile fits
// within the buffer.
static mlir::TypedValue<mlir::MemRefType> GetFullTileSubView(
    mlir::ImplicitLocOpBuilder& builder, TiledBufferInterface op) {
  auto offsets = mlir::getAsOpFoldResult(op.getOffsets());
  auto tile_size = GetStaticFoldResult(builder, op.getFullTileShape());
  auto strides = GetStaticFoldResult(builder, op.getStrides());

  mlir::RankedTensorType tile_type = op.getTile().getType();
  mlir::MemRefType subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          tile_type.getShape(), op.getBuffer().getType(), offsets, tile_size,
          strides);

  return mlir::memref::SubViewOp::create(builder, subview_type, op.getBuffer(),
                                         offsets, tile_size, strides);
}

// Get the subview of the local buffer - i.e it has 0 offsets & unit strides.
static mlir::TypedValue<mlir::MemRefType> GetLocalBufferSubview(
    mlir::ImplicitLocOpBuilder& builder,
    mlir::TypedValue<mlir::MemRefType> buffer,
    llvm::ArrayRef<mlir::OpFoldResult> tile_size,
    llvm::ArrayRef<int64_t> full_tile_shape) {
  mlir::SmallVector<mlir::OpFoldResult> buffer_offsets(
      buffer.getType().getRank(), builder.getIndexAttr(0));
  mlir::SmallVector<mlir::OpFoldResult> buffer_strides(
      buffer.getType().getRank(), builder.getIndexAttr(1));

  mlir::MemRefType buffer_subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          full_tile_shape, buffer.getType(), buffer_offsets, tile_size,
          buffer_strides);
  return mlir::memref::SubViewOp::create(builder, buffer_subview_type, buffer,
                                         buffer_offsets, tile_size,
                                         buffer_strides);
}

// Extract the slice of the tensor that is clamped to be within bounds of the
// target buffer.
static mlir::TypedValue<mlir::RankedTensorType> GetTensorSlice(
    mlir::ImplicitLocOpBuilder& builder, InsertTileOp op) {
  auto tile_size = GetClampedTileSize(builder, op, false);

  mlir::SmallVector<mlir::OpFoldResult> offsets(tile_size.size(),
                                                builder.getIndexAttr(0));
  mlir::SmallVector<mlir::OpFoldResult> strides(tile_size.size(),
                                                builder.getIndexAttr(1));

  return mlir::tensor::ExtractSliceOp::create(builder, op.getSource(), offsets,
                                              tile_size, strides);
}

static bool IsStaticallyAligned(mlir::Value value, int64_t alignment) {
  auto memref_type = mlir::cast<mlir::MemRefType>(value.getType());
  if (memref_type.getMemorySpaceAsInt() != 0) {
    return false;
  }

  // Check if it's a subview and try to prove alignment of dynamic offsets.
  if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    int64_t elem_bytes = memref_type.getElementTypeBitWidth() / 8;
    auto mixed_offsets = subview.getMixedOffsets();
    auto source_type =
        mlir::cast<mlir::MemRefType>(subview.getSource().getType());
    int64_t source_offset;
    llvm::SmallVector<int64_t> source_strides;
    if (mlir::failed(
            source_type.getStridesAndOffset(source_strides, source_offset))) {
      return false;
    }

    for (int i = 0; i < mixed_offsets.size(); ++i) {
      int64_t stride = source_strides[i];
      if (stride == mlir::ShapedType::kDynamic) {
        return false;
      }

      int64_t off_align;
      if (std::optional<int64_t> cst =
              mlir::getConstantIntValue(mixed_offsets[i])) {
        if (*cst == 0) {
          continue;
        }
        uint64_t ucst = std::abs(*cst);
        off_align = (ucst & -ucst);
      } else {
        off_align =
            GetKnownAlignment(mlir::cast<mlir::Value>(mixed_offsets[i]));
      }

      int64_t off_byte_align = off_align * stride * elem_bytes;
      if (off_byte_align % alignment != 0) {
        return false;
      }
    }
    return IsStaticallyAligned(subview.getSource(), alignment);
  }

  int64_t offset;
  llvm::SmallVector<int64_t> strides;
  if (mlir::failed(memref_type.getStridesAndOffset(strides, offset))) {
    return false;
  }

  if (offset == mlir::ShapedType::kDynamic || offset % alignment != 0) {
    return false;
  }

  if (!strides.empty() && (strides.back() == mlir::ShapedType::kDynamic ||
                           strides.back() % alignment != 0)) {
    return false;
  }

  return true;
}

static bool IsStaticallyFullTile(TiledBufferInterface op) {
  llvm::ArrayRef<int64_t> full_tile_shape = op.getFullTileShape();
  for (int64_t dim : full_tile_shape) {
    if (mlir::ShapedType::isDynamic(dim)) {
      return false;
    }
  }

  auto buffer_type = mlir::cast<mlir::MemRefType>(op.getBuffer().getType());
  if (!buffer_type.hasStaticShape()) {
    return false;
  }

  llvm::ArrayRef<int64_t> buffer_shape = buffer_type.getShape();
  if (buffer_shape.size() != full_tile_shape.size()) {
    return false;
  }

  // Check that offsets are in bounds.
  mlir::ValueRange offsets = op.getOffsets();
  llvm::ArrayRef<int64_t> strides = op.getStrides();
  for (int i = 0; i < offsets.size(); ++i) {
    int64_t stride_i = strides[i];
    std::optional<int64_t> offset = mlir::getConstantIntValue(offsets[i]);
    if (offset) {
      if (*offset + (full_tile_shape[i] - 1) * stride_i + 1 > buffer_shape[i]) {
        return false;
      }
      continue;
    }

    if (auto range = xla::GetRange(offsets[i])) {
      // If the range is within buffer bounds, this offset is statically full.
      if (range->lower >= 0 &&
          range->upper + (full_tile_shape[i] - 1) * stride_i + 1 <=
              buffer_shape[i]) {
        continue;
      }
    }
    return false;
  }

  return true;
}

static bool CanBeFullTile(TiledBufferInterface op) {
  auto buffer_type = mlir::cast<mlir::MemRefType>(op.getBuffer().getType());
  auto buffer_shape = buffer_type.getShape();
  auto tile_sizes = op.getFullTileShape();
  for (int i = 0; i < tile_sizes.size(); ++i) {
    if (buffer_shape[i] != mlir::ShapedType::kDynamic &&
        tile_sizes[i] > buffer_shape[i]) {
      return false;
    }
  }
  return true;
}

static bool IsContiguousLayout(mlir::MemRefType type) {
  if (type.getLayout().isIdentity()) {
    return true;
  }
  auto layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(type.getLayout());
  if (!layout) {
    return false;
  }

  auto shape = type.getShape();
  auto strides = layout.getStrides();
  int64_t expected = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (shape[i] == 1) {
      continue;
    }
    if (mlir::ShapedType::isDynamic(strides[i]) || strides[i] != expected) {
      return false;
    }
    if (mlir::ShapedType::isDynamic(shape[i])) {
      return false;
    }
    expected *= shape[i];
  }
  return true;
}
// Compute the subview result type without materializing an op in the IR.
static mlir::MemRefType GetFullTileSubViewType(TiledBufferInterface op) {
  mlir::OpBuilder builder(op.getContext());
  auto offsets = mlir::getAsOpFoldResult(op.getOffsets());
  auto tile_size = GetStaticFoldResult(builder, op.getFullTileShape());
  auto strides = GetStaticFoldResult(builder, op.getStrides());

  mlir::RankedTensorType tile_type = op.getTile().getType();
  return mlir::memref::SubViewOp::inferRankReducedResultType(
      tile_type.getShape(), op.getBuffer().getType(), offsets, tile_size,
      strides);
}

static mlir::Value TileIsFullSize(mlir::ImplicitLocOpBuilder& builder,
                                  TiledBufferInterface op) {
  if (IsStaticallyFullTile(op)) {
    return mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(),
                                              true);
  }

  llvm::SmallVector<mlir::OpFoldResult> clamped_tile_size =
      GetClampedTileSize(builder, op, false);
  mlir::Value is_full_size =
      mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(), true);
  for (auto [dim_idx, tile_dim_size] : llvm::enumerate(clamped_tile_size)) {
    if (auto value = tile_dim_size.dyn_cast<mlir::Value>()) {
      mlir::Value is_full_size_dim = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, value,
          mlir::arith::ConstantIndexOp::create(
              builder, op.getTile().getType().getDimSize(dim_idx)));
      is_full_size =
          mlir::arith::AndIOp::create(builder, is_full_size, is_full_size_dim);
    }
  }
  return is_full_size;
}

// Get a buffer copied from the original buffer that is padded to the full tile
// size.
static mlir::TypedValue<mlir::MemRefType> GetPaddedTileBuffer(
    mlir::ImplicitLocOpBuilder& builder, ExtractTileOp op) {
  auto buffer_tile_subview = GetClampedSubView(builder, op);
  mlir::RankedTensorType tile_type = op.getResult().getType();
  mlir::MemRefType subview_type = buffer_tile_subview.getType();
  int64_t offset;
  llvm::SmallVector<int64_t> strides;
  if (mlir::failed(subview_type.getStridesAndOffset(strides, offset))) {
    // If we can't get strided metadata, fallback to default layout.
    return mlir::memref::AllocOp::create(
        builder, GetStaticFoldResult(builder, tile_type.getShape()),
        tile_type.getElementType());
  }

  // Create a new contiguous buffer for the padded tile.
  auto buffer = mlir::memref::AllocOp::create(
      builder, GetStaticFoldResult(builder, tile_type.getShape()),
      tile_type.getElementType());

  auto local_tile_size = GetClampedTileSize(builder, op, false);
  auto local_buffer_subview =
      GetLocalBufferSubview(builder, buffer, local_tile_size,
                            buffer_tile_subview.getType().getShape());

  mlir::memref::CopyOp::create(builder, buffer_tile_subview,
                               local_buffer_subview);

  return buffer;
}

// Vector register file capacity threshold for deciding whether a tile should
// be forced into a local stack buffer. AVX-512 has 32 x 64-byte registers =
// 2048 bytes; we use 75% to leave headroom for intermediate values and
// loop-carried state.
static constexpr int64_t kVectorRegisterFileThresholdBytes = 1536;

// Returns true if `op` is a data-layout-altering operation such as transpose
// or broadcast. These operations have non-sequential memory access patterns
// that perform poorly when operating directly on strided buffer subviews
// (cache-line bouncing, register spill).
static bool IsLayoutAlteringOp(mlir::Operation* op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name.contains("transpose") || name.contains("broadcast") ||
      name.contains("reshape")) {
    return true;
  }

  // For linalg.generic, check if input indexing maps differ from the output
  // map, which indicates a data-layout transformation (e.g., the input is
  // accessed in transposed order relative to the output).
  if (name == "linalg.generic") {
    auto indexing_maps_attr =
        op->getAttrOfType<mlir::ArrayAttr>("indexing_maps");
    if (!indexing_maps_attr || indexing_maps_attr.size() < 2) {
      return false;
    }

    // The last map is the output; compare every input map against it.
    auto output_map =
        mlir::cast<mlir::AffineMapAttr>(indexing_maps_attr.getValue().back())
            .getValue();
    for (auto map_attr : indexing_maps_attr.getValue().drop_back()) {
      auto input_map = mlir::cast<mlir::AffineMapAttr>(map_attr).getValue();
      // A map with fewer results than the output indicates broadcast;
      // a map with the same results but different ordering indicates
      // transpose / permutation.
      if (input_map != output_map) {
        return true;
      }
    }
  }

  return false;
}

// Returns true if any direct user of `value` is a layout-altering op.
static bool HasLayoutAlteringUser(mlir::Value value) {
  for (mlir::Operation* user : value.getUsers()) {
    if (IsLayoutAlteringOp(user)) {
      return true;
    }
  }
  return false;
}

// Conservative estimate of a tile's memory footprint in bytes. Returns
// int64_t max for dynamic shapes so the caller treats them as "large".
static int64_t EstimateTileFootprintBytes(mlir::RankedTensorType type) {
  if (!type.hasStaticShape()) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t num_elements = 1;
  for (int64_t dim : type.getShape()) {
    num_elements *= dim;
  }
  int64_t element_bits = type.getElementTypeBitWidth();
  return num_elements * ((element_bits + 7) / 8);
}

// Decides whether `op` should force a contiguous local buffer allocation
// instead of yielding a (possibly strided) subview directly. This prevents
// performance regressions for:
//   1. Layout-altering ops (transpose, broadcast) that have non-sequential
//      access patterns on strided subviews → poor cache utilisation.
//   2. Large tiles that exceed the vector register file → register spill and
//      SROA (Scalar Replacement of Aggregates) failure.
static bool ShouldForceLocalBuffer(ExtractTileOp op) {
  return HasLayoutAlteringUser(op.getResult()) ||
         EstimateTileFootprintBytes(op.getResult().getType()) >
             kVectorRegisterFileThresholdBytes;
}

bool ExtractTileOp::bufferizesToMemoryRead(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

// ExtractTileOp is used to extract a tile from a buffer, so it does not
// bufferize to a memory write.
bool ExtractTileOp::bufferizesToMemoryWrite(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return false;
}

// ExtractTileOp bufferizes to an allocation if the tile is not statically full
// or if the buffer layout is not contiguous.
bool ExtractTileOp::bufferizesToAllocation(mlir::Value value) {
  return !IsStaticallyFullTile(*this) ||
         !IsContiguousLayout(GetFullTileSubViewType(*this)) ||
         ShouldForceLocalBuffer(*this);
}

mlir::bufferization::AliasingValueList ExtractTileOp::getAliasingValues(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return {{getResult(), mlir::bufferization::BufferRelation::Unknown}};
}

mlir::bufferization::AliasingOpOperandList ExtractTileOp::getAliasingOpOperands(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  DCHECK_EQ(value, getResult());
  mlir::bufferization::AliasingOpOperand result(
      &getSourceMutable(), mlir::bufferization::BufferRelation::Unknown, false);
  return {result};
}

bool ExtractTileOp::isWritable(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  return false;
}

llvm::LogicalResult ExtractTileOp::bufferize(
    mlir::RewriterBase& rewriter,
    const mlir::bufferization::BufferizationOptions& options,
    mlir::bufferization::BufferizationState& state) {
  mlir::ImplicitLocOpBuilder builder(getLoc(), rewriter);

  if (!CanBeFullTile(*this)) {
    auto buffer = GetPaddedTileBuffer(builder, *this);
    auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
        builder, getLoc(), getType(), buffer,
        /*restrict=*/true,
        /*writable=*/false);
    rewriter.replaceOp(getOperation(), to_tensor_op.getResult());
    return mlir::success();
  }

  auto subview_type = GetFullTileSubViewType(*this);

  const bool force_local_buffer = ShouldForceLocalBuffer(*this);

  if (IsStaticallyFullTile(*this) && IsContiguousLayout(subview_type) &&
      !force_local_buffer) {
    // Yield a subview directly; no copies or allocations required.
    auto subview = GetFullTileSubView(builder, *this);
    const int64_t alignment = xla::cpu::Align();
    const bool proved_alignment = IsStaticallyAligned(subview, alignment);
    if (proved_alignment) {
      builder.create<mlir::memref::AssumeAlignmentOp>(subview, alignment);
    }
    auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
        builder, getLoc(), getType(), subview,
        /*restrict=*/true,  // Depends on writeable=false.
        /*writable=*/false);

    if (proved_alignment) {
      for (auto user : getResult().getUsers()) {
        if (auto transfer_read =
                mlir::dyn_cast_or_null<mlir::vector::TransferReadOp>(user)) {
          transfer_read->setAttr("alignment",
                                 builder.getI64IntegerAttr(alignment));
        }
      }
    }

    rewriter.replaceOp(getOperation(), to_tensor_op.getResult());
    return mlir::success();
  }

  mlir::Value is_full_size = TileIsFullSize(builder, *this);
  int64_t alignment = xla::cpu::Align();

  // Compute identity strides based on full_tile_shape
  llvm::SmallVector<int64_t> identity_strides;
  int64_t stride = 1;
  auto full_tile_shape = getFullTileShape();
  for (int i = subview_type.getRank() - 1; i >= 0; --i) {
    identity_strides.push_back(stride);
    stride *= full_tile_shape[i];
  }
  std::reverse(identity_strides.begin(), identity_strides.end());

  auto yielded_layout = mlir::StridedLayoutAttr::get(
      getContext(), mlir::ShapedType::kDynamic, identity_strides);
  auto yielded_type = mlir::MemRefType::get(
      subview_type.getShape(), subview_type.getElementType(), yielded_layout);

  bool proved_alignment = false;
  auto if_op = mlir::scf::IfOp::create(
      builder, is_full_size,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder then_builder(loc, builder);
        auto buffer = GetFullTileSubView(then_builder, *this);
        mlir::Value yielded_buffer = buffer;
        if (!IsContiguousLayout(buffer.getType()) || force_local_buffer) {
          llvm::errs() << "Force local buffer, " << buffer.getType() << "\n";
          llvm::errs() << *this << "\n";
          llvm::errs() << "Force local buffer: " << force_local_buffer << "\n";
          // Force allocation to obtain a contiguous buffer.
          auto contiguous_type = mlir::MemRefType::get(
              buffer.getType().getShape(), buffer.getType().getElementType());
          auto alloc =
              then_builder.create<mlir::memref::AllocOp>(contiguous_type);
          then_builder.create<mlir::memref::CopyOp>(buffer, alloc);
          yielded_buffer = alloc;
        }
        if (IsStaticallyAligned(yielded_buffer, alignment)) {
          then_builder.create<mlir::memref::AssumeAlignmentOp>(yielded_buffer,
                                                               alignment);
          proved_alignment = true;
        }
        auto cast = then_builder.create<mlir::memref::CastOp>(yielded_type,
                                                              yielded_buffer);
        then_builder.create<mlir::scf::YieldOp>(cast.getResult());
      },

      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder else_builder(loc, builder);
        auto buffer = GetPaddedTileBuffer(else_builder, *this);
        else_builder.create<mlir::memref::AssumeAlignmentOp>(buffer, alignment);
        auto cast =
            else_builder.create<mlir::memref::CastOp>(yielded_type, buffer);
        else_builder.create<mlir::scf::YieldOp>(cast.getResult());
      });

  auto if_res = if_op.getResult(0);
  if (proved_alignment) {
    builder.create<mlir::memref::AssumeAlignmentOp>(if_res, alignment);
  }

  mlir::bufferization::ToTensorOp to_tensor_op =
      mlir::bufferization::ToTensorOp::create(builder, getType(), if_res);
  to_tensor_op.setRestrict(true);

  if (proved_alignment) {
    for (auto user : getResult().getUsers()) {
      if (auto transfer_read =
              mlir::dyn_cast_or_null<mlir::vector::TransferReadOp>(user)) {
        transfer_read->setAttr("alignment",
                               builder.getI64IntegerAttr(alignment));
      }
    }
  }

  rewriter.replaceOp(getOperation(), to_tensor_op.getResult());

  return mlir::success();
}

bool InsertTileOp::bufferizesToMemoryWrite(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  DCHECK(mlir::isa<mlir::RankedTensorType>(operand.get().getType()))
      << "This should only be called on the tensor operand.";
  return false;
}

bool InsertTileOp::bufferizesToAllocation(mlir::Value value) { return false; }

mlir::bufferization::AliasingValueList InsertTileOp::getAliasingValues(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return {};
}

mlir::bufferization::AliasingOpOperandList InsertTileOp::getAliasingOpOperands(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  return {};
}

bool InsertTileOp::bufferizesToMemoryRead(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool InsertTileOp::isWritable(mlir::Value value,
                              const mlir::bufferization::AnalysisState& state) {
  return value == getDestination();
}

llvm::LogicalResult InsertTileOp::bufferize(
    mlir::RewriterBase& rewriter,
    const mlir::bufferization::BufferizationOptions& options,
    mlir::bufferization::BufferizationState& state) {
  mlir::ImplicitLocOpBuilder builder(getLoc(), rewriter);

  if (!CanBeFullTile(*this)) {
    auto tile_slice = GetTensorSlice(builder, *this);
    auto target_buffer_subview = GetClampedSubView(builder, *this);
    auto materialize_op =
        mlir::bufferization::MaterializeInDestinationOp::create(
            builder, tile_slice, target_buffer_subview);
    materialize_op.setWritable(true);
    rewriter.eraseOp(getOperation());
    return mlir::success();
  }

  if (IsStaticallyFullTile(*this)) {
    auto buffer = GetFullTileSubView(builder, *this);

    int64_t alignment = xla::cpu::Align();
    if (auto transfer_write =
            mlir::dyn_cast_or_null<mlir::vector::TransferWriteOp>(
                getSource().getDefiningOp())) {
      if (mlir::isa_and_nonnull<mlir::tensor::EmptyOp>(
              transfer_write.getBase().getDefiningOp())) {
        if (IsContiguousLayout(buffer.getType()) &&
            IsStaticallyAligned(buffer, alignment)) {
          builder.create<mlir::memref::AssumeAlignmentOp>(buffer, alignment);
          auto new_transfer_write =
              builder.create<mlir::vector::TransferWriteOp>(
                  transfer_write.getVector(), buffer,
                  transfer_write.getIndices(),
                  transfer_write.getPermutationMapAttr(),
                  transfer_write.getInBoundsAttr());
          if (auto attr = transfer_write->getAttr("alignment")) {
            new_transfer_write.getOperation()->setAttr("alignment", attr);
          } else {
            new_transfer_write.getOperation()->setAttr(
                "alignment", builder.getI64IntegerAttr(alignment));
          }
          rewriter.eraseOp(getOperation());
          return mlir::success();
        }
      }
    }

    if (IsStaticallyAligned(buffer, alignment)) {
      builder.create<mlir::memref::AssumeAlignmentOp>(buffer, alignment);
    }
    builder.create<mlir::bufferization::MaterializeInDestinationOp>(getSource(),
                                                                    buffer);
    rewriter.eraseOp(getOperation());
    return mlir::success();
  }
  mlir::Value is_full_size = TileIsFullSize(builder, *this);

  mlir::scf::IfOp::create(
      builder, is_full_size,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder then_builder(loc, builder);
        auto target_buffer_subview = GetFullTileSubView(then_builder, *this);
        auto materialize_op =
            mlir::bufferization::MaterializeInDestinationOp::create(
                then_builder, getSource(), target_buffer_subview);
        materialize_op.setWritable(true);
        mlir::scf::YieldOp::create(then_builder);
      },
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder else_builder(loc, builder);
        auto tile_slice = GetTensorSlice(else_builder, *this);
        auto target_buffer_subview = GetClampedSubView(else_builder, *this);
        auto materialize_op =
            mlir::bufferization::MaterializeInDestinationOp::create(
                else_builder, tile_slice, target_buffer_subview);
        materialize_op.setWritable(true);
        mlir::scf::YieldOp::create(else_builder);
      });

  rewriter.eraseOp(getOperation());
  return mlir::success();
}

}  // namespace xla::xtile

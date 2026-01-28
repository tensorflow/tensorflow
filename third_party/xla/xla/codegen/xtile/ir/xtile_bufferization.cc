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
#include "xla/codegen/emitters/implicit_arith_op_builder.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::xtile {

static llvm::SmallVector<mlir::OpFoldResult> GetStaticFoldResult(
    mlir::OpBuilder& builder, llvm::ArrayRef<int64_t> input) {
  return llvm::map_to_vector(input, [&builder](int64_t value) {
    return mlir::OpFoldResult(builder.getIndexAttr(value));
  });
}

static llvm::SmallVector<mlir::OpFoldResult> GetDynamicFoldResult(
    mlir::ValueRange input) {
  return llvm::SmallVector<mlir::OpFoldResult>(input);
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

  auto offsets = GetDynamicFoldResult(op.getOffsets());
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
  auto offsets = GetDynamicFoldResult(op.getOffsets());
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

static mlir::Value TileIsFullSize(mlir::ImplicitLocOpBuilder& builder,
                                  TiledBufferInterface op) {
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

bool ExtractTileOp::bufferizesToMemoryRead(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool ExtractTileOp::bufferizesToMemoryWrite(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool ExtractTileOp::bufferizesToAllocation(mlir::Value value) {
  // As we don't know if we will emit an allocation at compile time we must be
  // conservative.
  return true;
}

mlir::bufferization::AliasingValueList ExtractTileOp::getAliasingValues(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return {};
}

mlir::bufferization::AliasingOpOperandList ExtractTileOp::getAliasingOpOperands(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  DCHECK_EQ(value, getResult());
  mlir::bufferization::AliasingOpOperand result(
      &getSourceMutable(), mlir::bufferization::BufferRelation::Equivalent,
      false);
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

  mlir::Value is_full_size = TileIsFullSize(builder, *this);
  auto if_op = mlir::scf::IfOp::create(
      builder, is_full_size,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder then_builder(loc, builder);
        auto buffer = GetFullTileSubView(then_builder, *this);
        if (buffer.getType().getLayout().isIdentity()) {
          auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
              then_builder, getType(), buffer);
          mlir::scf::YieldOp::create(then_builder, {to_tensor_op});
        } else {
          // If the buffer doesn't have an identity layout, we can get a
          // miscompile during bufferization as some ops don't support
          // non-identity layouts. So we allocate a new buffer with the same
          // shape but default layout.
          // TODO(willfroom): Look into how we can remove this constraint.
          mlir::MemRefType default_buffer_type =
              mlir::MemRefType::Builder(buffer.getType()).setLayout(nullptr);
          auto default_buffer =
              mlir::memref::AllocOp::create(then_builder, default_buffer_type);
          mlir::memref::CopyOp::create(then_builder, buffer, default_buffer);
          auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
              then_builder, getType(), default_buffer);
          to_tensor_op.setWritable(true);
          to_tensor_op.setRestrict(true);
          mlir::scf::YieldOp::create(then_builder, {to_tensor_op});
        }
      },
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder else_builder(loc, builder);
        auto buffer = GetPaddedTileBuffer(else_builder, *this);
        auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
            else_builder, getType(), buffer);
        to_tensor_op.setWritable(true);
        to_tensor_op.setRestrict(true);
        mlir::scf::YieldOp::create(else_builder, {to_tensor_op});
      });

  rewriter.replaceOp(getOperation(), if_op.getResults());

  return mlir::success();
}

bool InsertTileOp::bufferizesToMemoryRead(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool InsertTileOp::bufferizesToMemoryWrite(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  DCHECK_EQ(operand.getOperandNumber(), 0)
      << "This should only be called on the tensor operand.";
  return false;
}

bool InsertTileOp::bufferizesToAllocation(mlir::Value value) {
  // As we don't know if we will emit an allocation at compile time we must be
  // conservative.
  return true;
}

mlir::bufferization::AliasingValueList InsertTileOp::getAliasingValues(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return {};
}

mlir::bufferization::AliasingOpOperandList InsertTileOp::getAliasingOpOperands(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  return {};
}

bool InsertTileOp::isWritable(mlir::Value value,
                              const mlir::bufferization::AnalysisState& state) {
  if (value == getDestination()) {
    return true;
  }

  return false;
}

llvm::LogicalResult InsertTileOp::bufferize(
    mlir::RewriterBase& rewriter,
    const mlir::bufferization::BufferizationOptions& options,
    mlir::bufferization::BufferizationState& state) {
  mlir::ImplicitLocOpBuilder builder(getLoc(), rewriter);

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

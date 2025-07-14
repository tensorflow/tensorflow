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

#include "xla/mosaic/dialect/tpu/transforms/infer_memref_layout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "absl/log/check.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_INFERMEMREFLAYOUTPASS
#define GEN_PASS_DEF_INFERMEMREFLAYOUTPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

// Returns the number of lanes (usually 128) groups in a tile.
//
// Arguments:
//   src_sublane: A number of lanes in the full operand.
//   hardware_generation: An integer indicating the target TPU generation.
//   target_sublane_count: The number of sublane in the target shape.
//   tpu_tiling_flags: A struct of flags indicating which large tiling modes are
//     enabled by XLA for memrefs.
//   bitwidth: The bitwidth of the element type of the operand.
//   is_kernel_argument: Whether the operand is a kernel argument.
//   is_1d: Whether the operand is 1D.
int getTilingFactor(const int src_sublane, const int hardware_generation,
                    const int64_t target_sublane_count,
                    const TpuTilingFlags &tpu_tiling_flags,
                    const int8_t bitwidth, const bool is_kernel_argument,
                    const bool is_1d) {
  CHECK(llvm::isPowerOf2_32(bitwidth));
  CHECK_LE(2, bitwidth);
  CHECK_LE(bitwidth, 32);
  const int packing = 32 / bitwidth;
  const int min_tiling = (1 + (hardware_generation < 4)) * packing;
  // When packing is larger than the sublane count, we want its tiling to be at
  // least as large as the packing to make sure we can fully pack values. For
  // example, for int2 on the target with 8 sublanes, we want the tiling to be
  // at least 16.
  const int64_t tiling_sublane =
      std::max(target_sublane_count, static_cast<int64_t>(packing));
  const int max_normal_tiling = tiling_sublane;

  int large_tiling = [&] {
    if (is_1d) {
      // 1D tiling is always compact.
      return tiling_sublane;
    }
    if (bitwidth == 2) {
      return target_sublane_count * 16;
    }
    if (bitwidth == 4 && tpu_tiling_flags.use_x4_large_second_minor) {
      return target_sublane_count * 8;
    }
    if (bitwidth == 8 && tpu_tiling_flags.use_x8_large_second_minor) {
      return target_sublane_count * 4;
    }
    // 16-bit values are generally always possible to relayout on the fly in v6,
    // so we allow large 2nd minor tiling whenever possible. We can't do this
    // for kernel arguments, because the layout of those is controlled by XLA.
    if (bitwidth == 16 && (tpu_tiling_flags.use_x16_large_second_minor ||
                           (!is_kernel_argument && hardware_generation >= 6))) {
      return target_sublane_count * 2;
    }
    return tiling_sublane;
  }();

  bool is_divisible = src_sublane % large_tiling == 0;
  large_tiling = is_divisible ? large_tiling : tiling_sublane;

  // Use large tiling if our operand is tall enough to fit at least one full
  // tile.
  if (large_tiling <= src_sublane) {
    return large_tiling;
  }

  int tiling = min_tiling;
  while (tiling < std::min(src_sublane, max_normal_tiling)) {
    tiling *= 2;
  }
  return tiling;
}

FailureOr<TiledLayoutAttr> inferLayout(MemRefType memref_ty,
                                       const int hardware_generation,
                                       std::array<int64_t, 2> target_shape,
                                       const TpuTilingFlags &tpu_tiling_flags,
                                       bool is_kernel_argument,
                                       int64_t leading_tile_rows = 0) {
  if (auto tiled_layout_attr =
          dyn_cast<TiledLayoutAttr>(memref_ty.getLayout())) {
    if (leading_tile_rows > 0 && !tiled_layout_attr.getTiles().empty() &&
        tiled_layout_attr.getTiles().front().dimensions().size() == 2 &&
        tiled_layout_attr.getTiles().front().dimensions()[0] !=
            leading_tile_rows) {
      return emitError(UnknownLoc::get(memref_ty.getContext()),
                       "Trying to infer memref layout with sublane tiling ")
             << leading_tile_rows
             << ", but the memref already has sublane tiling "
             << tiled_layout_attr.getTiles().front().dimensions()[0];
    }
    return tiled_layout_attr;
  }
  if (auto affine_map_attr = dyn_cast<AffineMapAttr>(memref_ty.getLayout())) {
    if (!affine_map_attr.isIdentity()) {
      return emitError(UnknownLoc::get(memref_ty.getContext()),
                       "Non-identity affine layout");
    }
  } else if (!isa<StridedLayoutAttr>(memref_ty.getLayout())) {
    return emitError(UnknownLoc::get(memref_ty.getContext()),
                     "Unrecognized layout annotation");
  }
  if (memref_ty.getRank() == 0) {
    return emitError(UnknownLoc::get(memref_ty.getContext()),
                     "0-rank memref not supported");
  }
  if (!memref_ty.getElementType().isIntOrFloat()) {
    return emitError(UnknownLoc::get(memref_ty.getContext()),
                     "Invalid element type for memref");
  }
  const int8_t bitwidth = memref_ty.getElementTypeBitWidth();
  const auto [sublane_count, lane_count] = target_shape;
  // Infer the layout
  if (memref_ty.getRank() == 1) {
    auto src_sublane =
        llvm::divideCeil(memref_ty.getShape().back(), lane_count);
    const int64_t leading_tile =
        getTilingFactor(src_sublane, hardware_generation, sublane_count,
                        tpu_tiling_flags, bitwidth, is_kernel_argument,
                        /*is_1d=*/true) *
        lane_count;
    SmallVector<xla::Tile> tiles{xla::Tile({leading_tile})};
    if (bitwidth != 32) {
      if (!llvm::has_single_bit<unsigned>(bitwidth) || bitwidth > 32) {
        return emitError(UnknownLoc::get(memref_ty.getContext()),
                         "Unsupported bitwidth: ")
               << bitwidth;
      }
      tiles.append({xla::Tile({lane_count}), xla::Tile({32 / bitwidth, 1})});
    }
    return TiledLayoutAttr::get(memref_ty.getContext(), tiles, {1});
  }

  // memref.getRank() > 1
  const ArrayRef<int64_t> shape = memref_ty.getShape();

  const int64_t src_sublane = shape[shape.size() - 2];
  if (leading_tile_rows == 0) {
    leading_tile_rows = getTilingFactor(
        src_sublane, hardware_generation, sublane_count, tpu_tiling_flags,
        bitwidth, is_kernel_argument, /*is_1d=*/false);
  }
  SmallVector<xla::Tile> tiles{xla::Tile({leading_tile_rows, lane_count})};
  if (bitwidth != 32) {
    if (!llvm::has_single_bit<unsigned>(bitwidth) || bitwidth > 32) {
      return emitError(UnknownLoc::get(memref_ty.getContext()),
                       "Unsupported bitwidth: ")
             << bitwidth;
    }
    tiles.push_back(xla::Tile({32 / bitwidth, 1}));
  }
  auto tile_strides =
      ComputeTileStrides(memref_ty, {leading_tile_rows, lane_count});
  return TiledLayoutAttr::get(memref_ty.getContext(), tiles, tile_strides);
}

// Make sure only the first tile might introduce padding.
LogicalResult checkTiles(MLIRContext *mlir_ctx,
                         const ArrayRef<xla::Tile> &tiles) {
  SmallVector<int64_t> tiled_dims(tiles.front().dimensions().begin(),
                                  tiles.front().dimensions().end());
  for (const xla::Tile &t : tiles.drop_front()) {
    const int64_t offset = tiled_dims.size() - t.dimensions().size();
    if (offset < 0) {
      return emitError(UnknownLoc::get(mlir_ctx),
                       "Not implemented: layout too complicated");
    }
    for (int i = 0; i < t.dimensions().size(); ++i) {
      auto [d, m] = std::div(tiled_dims[offset + i], t.dimension(i));
      if (m != 0) {
        return emitError(UnknownLoc::get(mlir_ctx),
                         "Not implemented: layout too complicated");
      }
      tiled_dims[offset + i] = d;
    }
    tiled_dims.append(t.dimensions().begin(), t.dimensions().end());
  }
  return success();
}

FailureOr<MemRefType> inferMemref(MemRefType memref,
                                  const int hardware_generation,
                                  std::array<int64_t, 2> target_shape,
                                  const TpuTilingFlags &tpu_tiling_flags,
                                  bool is_kernel_argument,
                                  int64_t leading_tile_rows) {
  if (isa<SemaphoreType, DMASemaphoreType>(memref.getElementType())) {
    const Attribute semaphore_mem = tpu::MemorySpaceAttr::get(
        memref.getContext(), MemorySpace::kSemaphoreMem);
    SmallVector<int64_t> tile_strides;
    tile_strides.reserve(memref.getRank());
    int64_t stride = 1;
    for (int i = memref.getRank() - 1; i >= 0; --i) {
      tile_strides.push_back(stride);
      stride *= memref.getDimSize(i);
    }
    std::reverse(tile_strides.begin(), tile_strides.end());
    auto layout = TiledLayoutAttr::get(memref.getContext(), {}, tile_strides);
    return MemRefType::get(memref.getShape(), memref.getElementType(), layout,
                           semaphore_mem);
  }
  const Attribute vmem =
      tpu::MemorySpaceAttr::get(memref.getContext(), MemorySpace::kVmem);
  const Attribute memory_space =
      memref.getMemorySpace() == nullptr ? vmem : memref.getMemorySpace();
  FAILUREOR_ASSIGN_OR_RETURN(
      const TiledLayoutAttr layout,
      inferLayout(memref, hardware_generation, target_shape, tpu_tiling_flags,
                  is_kernel_argument, leading_tile_rows));

  const ArrayRef<xla::Tile> tiles = layout.getTiles();
  if (failed(checkTiles(memref.getContext(), tiles))) {
    return failure();
  }
  const xla::Tile &first_tile = tiles.front();
  const int64_t untiled_dims =
      memref.getShape().size() - first_tile.dimensions().size();
  if (untiled_dims < 0) {
    return emitError(UnknownLoc::get(memref.getContext()), "Invalid tiling");
  }
  SmallVector<int64_t> new_shape(memref.getShape());
  for (int i = 0; i < first_tile.dimensions().size(); ++i) {
    new_shape[untiled_dims + i] =
        llvm::alignTo(new_shape[untiled_dims + i], first_tile.dimension(i));
  }
  return MemRefType::get(new_shape, memref.getElementType(), layout,
                         memory_space);
}

LogicalResult inferOp(Operation &op, const int hardware_generation,
                      std::array<int64_t, 2> target_shape,
                      const TpuTilingFlags &tpu_tiling_flags) {
  if (auto alloca_op = dyn_cast<memref::AllocaOp>(op)) {
    TypedValue<MemRefType> arg = alloca_op.getResult();
    const MemRefType memref_ty = alloca_op.getResult().getType();
    // If the memref can be reinterpreted to untiled, force to use tiling
    // {1, target.lane_count} for 32 bit.
    int64_t leading_tile_rows = 0;
    // TODO(b/375038685): generalize untiled memref with packed type which
    // needs to update load/store rules.
    if (memref_ty.getElementTypeBitWidth() == 32 && memref_ty.getRank() > 1 &&
        *(memref_ty.getShape().end() - 1) <= target_shape[1]) {
      leading_tile_rows = 1;
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const MemRefType new_memref_ty,
        inferMemref(memref_ty, hardware_generation, target_shape,
                    tpu_tiling_flags, /*is_kernel_argument=*/false,
                    leading_tile_rows));
    alloca_op.getResult().setType(new_memref_ty);
    if (memref_ty != new_memref_ty) {
      OpBuilder builder(alloca_op->getContext());
      builder.setInsertionPointAfter(alloca_op);
      // TODO(b/376130272): add a canonicalizer for EraseLayoutOp so that if we
      // have erase(erase(x)) then we rewrite it to erase(x).
      auto erase_op = builder.create<tpu::EraseLayoutOp>(
          arg.getLoc(),
          MemRefType::get(new_memref_ty.getShape(), memref_ty.getElementType(),
                          /*layout=*/nullptr, new_memref_ty.getMemorySpace()),
          arg);
      arg.replaceAllUsesExcept(erase_op.getResult(), erase_op);
    }
  } else if (auto alloca_op = dyn_cast<tpu::AllocaSemaphoreOp>(op)) {
    TypedValue<MemRefType> arg = alloca_op.getResult();
    const MemRefType memref_ty = alloca_op.getResult().getType();
    FAILUREOR_ASSIGN_OR_RETURN(
        const MemRefType new_memref_ty,
        inferMemref(memref_ty, hardware_generation, target_shape,
                    tpu_tiling_flags, /*is_kernel_argument=*/false));
    alloca_op.getResult().setType(new_memref_ty);
    if (memref_ty != new_memref_ty) {
      OpBuilder builder(alloca_op->getContext());
      builder.setInsertionPointAfter(alloca_op);
      auto erase_op = builder.create<tpu::EraseLayoutOp>(
          arg.getLoc(),
          MemRefType::get(new_memref_ty.getShape(), memref_ty.getElementType(),
                          /*layout=*/nullptr, new_memref_ty.getMemorySpace()),
          arg);
      arg.replaceAllUsesExcept(erase_op.getResult(), erase_op);
    }
  }
  for (Region &region : op.getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (failed(inferOp(op, hardware_generation, target_shape,
                           tpu_tiling_flags))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult inferFunc(func::FuncOp f, const int hardware_generation,
                        std::array<int64_t, 2> target_shape,
                        const TpuTilingFlags &tpu_tiling_flags) {
  if (!f.getBody().hasOneBlock()) {
    return f.emitOpError("Functions should only have a single block");
  }
  Block &entry = f.getBody().front();
  SmallVector<Type> new_arg_types;
  auto builder = OpBuilder::atBlockBegin(&entry);
  for (int i = 0; i < entry.getNumArguments(); ++i) {
    BlockArgument arg = entry.getArgument(i);
    const auto memref_ty = dyn_cast<MemRefType>(arg.getType());
    if (memref_ty == nullptr) {
      new_arg_types.push_back(arg.getType());
      continue;
    }
    int64_t leading_tile_rows = 0;
    auto leading_tile_rows_attr =
        f.getArgAttrOfType<mlir::IntegerAttr>(i, kLeadingTileRows);
    if (leading_tile_rows_attr != nullptr) {
      leading_tile_rows = leading_tile_rows_attr.getInt();
      f.removeArgAttr(i, kLeadingTileRows);
    }

    FAILUREOR_ASSIGN_OR_RETURN(
        MemRefType new_memref_ty,
        inferMemref(memref_ty, hardware_generation, target_shape,
                    tpu_tiling_flags, /*is_kernel_argument=*/true,
                    leading_tile_rows));
    arg.setType(new_memref_ty);
    new_arg_types.push_back(arg.getType());
    if (memref_ty != new_memref_ty) {
      Value val = arg;
      Operation *arg_use_op = nullptr;
      // If the arg memref can be reinterpreted to untiled, we can insert
      // ReinterpretCastOp to use tiling {packing, target.lane_count} before
      // EraseLayoutOp for only the arg memrefs and expect the rest memref
      // layout inference is based on the casted layout automatically. This
      // would help lift many restrictions in alignment check when consuming
      // this memref.
      if (canReinterpretToUntiledMemref(cast<TypedValue<MemRefType>>(val),
                                        target_shape,
                                        /*allow_minormost_padding=*/true) &&
          // TODO(b/375038685): generalize untiled memref with packed type which
          // needs to update load/store rules.
          new_memref_ty.getElementTypeBitWidth() == 32) {
        auto tiled_layout =
            cast<tpu::TiledLayoutAttr>(new_memref_ty.getLayout());
        SmallVector<xla::Tile> tiles(tiled_layout.getTiles());
        SmallVector<int64_t> new_tile_strides(tiled_layout.getTileStrides());
        for (int i = 0; i < new_tile_strides.size() - 2; ++i) {
          new_tile_strides[i] *= tiles[0].dimension(0);
        }
        tiles[0] = ::xla::Tile({1, target_shape[1]});
        new_memref_ty = MemRefType::get(
            new_memref_ty.getShape(), new_memref_ty.getElementType(),
            TiledLayoutAttr::get(new_memref_ty.getContext(), tiles,
                                 new_tile_strides),
            new_memref_ty.getMemorySpace());
        arg_use_op = builder.create<tpu::ReinterpretCastOp>(val.getLoc(),
                                                            new_memref_ty, val);
        val = arg_use_op->getResult(0);
      }
      // Some standard MLIR ops have static checks that seems unreasonable,
      // and we know they hold in the way they are used in Mosaic. Still,
      // verification with layouts likes to fail, because it can't statically
      // prove the properties.
      auto erase_op = builder.create<tpu::EraseLayoutOp>(
          val.getLoc(),
          MemRefType::get(new_memref_ty.getShape(), memref_ty.getElementType(),
                          /*layout=*/nullptr, new_memref_ty.getMemorySpace()),
          val);
      if (!arg_use_op) {
        arg_use_op = erase_op;
      }
      arg.replaceAllUsesExcept(erase_op.getResult(), arg_use_op);
    }
  }
  f.setFunctionType(
      builder.getAttr<FunctionType>(new_arg_types, f.getResultTypes()));
  for (Operation &op : entry.getOperations()) {
    if (failed(
            inferOp(op, hardware_generation, target_shape, tpu_tiling_flags))) {
      return failure();
    }
  }
  return success();
}

struct InferMemRefLayoutPass
    : public impl::InferMemRefLayoutPassBase<InferMemRefLayoutPass> {
  InferMemRefLayoutPass(int hardware_generation_,
                        std::array<int64_t, 2> target_shape_,
                        const TpuTilingFlags &tpu_tiling_flags_) {
    hardware_generation = hardware_generation_;
    sublane_count = target_shape_[0];
    lane_count = target_shape_[1];
    tpu_tiling_flags = tpu_tiling_flags_;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    if (failed(inferFunc(func, hardware_generation, {sublane_count, lane_count},
                         tpu_tiling_flags))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createInferMemRefLayoutPass(
    int hardware_generation, std::array<int64_t, 2> target_shape,
    const TpuTilingFlags &tpu_tiling_flags_) {
  return std::make_unique<InferMemRefLayoutPass>(
      hardware_generation, target_shape, tpu_tiling_flags_);
}

}  // namespace mlir::tpu

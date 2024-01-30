/*Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/fusions/tiling_util.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

void EmitTileRec(const TilingThreadIdInfo& thread_id_info,
                 const TilingScheme& tiling_scheme, int dim,
                 std::array<llvm::Value*, 3> tile_idx,
                 absl::Span<llvm::Value* const> tile_dimensions,
                 llvm::IRBuilder<>* b, const TileElementGenerator& emit_elem) {
  llvm::Type* index_ty = thread_id_info.thread_id->getType();
  auto constant = [&](int64_t val) {
    return llvm::ConstantInt::get(index_ty, val);
  };

  auto recurse = [&] {
    EmitTileRec(thread_id_info, tiling_scheme, dim + 1, tile_idx,
                tile_dimensions, b, emit_elem);
  };

  KernelSupportLibrary ksl(b, dim == TilingScheme::DimX
                                  ? llvm_ir::UnrollMode::kFullyUnroll
                                  : llvm_ir::UnrollMode::kDefaultUnroll);

  // TODO(jreiffers): Remove the dim==Z check, this is only here for historical
  // reasons.
  if (dim == TilingScheme::DimZ && tiling_scheme.GetBlockTileSize()[dim] == 1) {
    tile_idx[dim] = constant(0);
    recurse();
  } else if (dim == TilingScheme::DimX) {
    int64_t vector_size = tiling_scheme.GetVectorSize();
    int64_t stride = tiling_scheme.GetThreadsPerBlock()[TilingScheme::DimX];
    int64_t last_dim_size = tiling_scheme.GetThreadTileSize()[2] / vector_size;

    auto make_loop = [&](bool emit_bounds_checks) {
      auto body = [&, emit_bounds_checks](llvm::Value* i) {
        for (int64_t v = 0; v < vector_size; ++v) {
          tile_idx[dim] = b->CreateAdd(
              b->CreateAdd(b->CreateMul(i, constant(stride * vector_size)),
                           constant(v)),
              thread_id_info.start_offsets[dim]);
          if (emit_bounds_checks) {
            auto* in_bounds =
                b->CreateICmpULT(tile_idx[dim], tile_dimensions[dim]);
            ksl.If("x_in_tile", in_bounds, [&] { emit_elem(tile_idx); });
          } else {
            emit_elem(tile_idx);
          }
        }
      };
      return [&, body] {
        ksl.For(absl::StrCat("loop", dim), constant(0), constant(last_dim_size),
                constant(1), body);
      };
    };
    if (stride > 1 && last_dim_size > 1) {
      // Most tiles will be full, so we emit a single bounds check for those.
      auto* is_full_tile =
          b->CreateICmpEQ(constant(tiling_scheme.GetBlockTileSize()[dim]),
                          tile_dimensions[dim]);
      ksl.If("is_full_tile", is_full_tile, make_loop(false), make_loop(true));
    } else {
      // TODO(jreiffers): If last_dim_size is 1, we don't need the bounds check
      // and actually we don't need any loop. That's a special case of the TODO
      // above.
      make_loop(true)();
    }
  } else {
    ksl.For(absl::StrCat("loop", dim), thread_id_info.start_offsets[dim],
            tile_dimensions[dim],
            constant(tiling_scheme.GetThreadsPerBlock()[dim]),
            [&](llvm::Value* i) {
              tile_idx[dim] = i;
              recurse();
            });
  }
}

}  // namespace

void EmitTile(llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
              const TilingThreadIdInfo& thread_id_info,
              absl::Span<llvm::Value* const> tile_dimensions,
              const TileElementGenerator& emit_elem_function) {
  EmitTileRec(thread_id_info, tiling_scheme, 0, {}, tile_dimensions, builder,
              emit_elem_function);
}

namespace {

// Emits current block id.
llvm::Value* EmitBlockId(llvm::IRBuilder<>* builder, int32_t num_blocks,
                         llvm::Type* index_ty) {
  llvm::Value* block_id =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, builder);
  if (num_blocks != 0) {
    llvm_ir::AddRangeMetadata(0, num_blocks,
                              llvm::cast<llvm::Instruction>(block_id));
  }
  return builder->CreateIntCast(block_id, index_ty, /*isSigned=*/true,
                                "block.id.x");
}

// Emits current thread id with the given type.
//
// Sets the return value range to [0, threads_per_block).
llvm::Value* EmitThreadId(llvm::IRBuilder<>* builder, int64_t threads_per_block,
                          llvm::Type* index_ty) {
  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::CallInst* thread_id_raw =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, builder);
  llvm_ir::AddRangeMetadata(0, threads_per_block, thread_id_raw);
  return builder->CreateIntCast(thread_id_raw, index_ty,
                                /*isSigned=*/true, "thread.id.x");
}

// Emits the LLVM values for thread_id, block_id, coordinates of the current
// tile and strides of the loops to iterate over the current tile.
absl::StatusOr<TilingThreadIdInfo> EmitThreadIdInfo(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* index_ty) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  llvm::Value* thread_id =
      EmitThreadId(builder, tiling_scheme.GetNumThreadsPerBlock(), index_ty);
  int64_t num_blocks = tiling_scheme.GetNumBlocks();
  if (num_blocks > (int64_t)std::numeric_limits<uint32_t>::max()) {
    return FailedPrecondition(
        "Number of physical blocks (%d) does not fit in an i32 in tiling "
        "scheme: %s",
        num_blocks, tiling_scheme.ToString());
  }
  llvm::Value* block_id = EmitBlockId(builder, num_blocks, index_ty);

  auto num_threads = tiling_scheme.GetThreadsPerBlock();
  auto* stride_y = constant(num_threads[TilingScheme::DimX]);
  auto* stride_z = constant(num_threads[TilingScheme::DimY] *
                            num_threads[TilingScheme::DimX]);

  auto* thread_id_z = builder->CreateUDiv(thread_id, stride_z, "thread_id.z");
  auto* thread_id_y = builder->CreateUDiv(
      builder->CreateURem(thread_id, stride_z), stride_y, "thread_id.y");
  auto* thread_id_x = builder->CreateURem(thread_id, stride_y, "thread_id.x");

  std::array<llvm::Value*, 3> thread_ids{thread_id_z, thread_id_y, thread_id_x};
  std::array<llvm::Value*, 3> start_offsets{
      thread_id_z, thread_id_y,
      builder->CreateMul(thread_id_x, constant(tiling_scheme.GetVectorSize()))};

  auto* lane_id =
      builder->CreateURem(thread_id, constant(WarpSize()), "lane_id");
  return TilingThreadIdInfo{thread_id, thread_ids, start_offsets, lane_id,
                            block_id};
}

}  // namespace

absl::StatusOr<TilingKernelInfo> EmitTilingKernel(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* index_ty, const TileGenerator& tile_generator) {
  absl::Span<const int64_t> dims_in_elems = tiling_scheme.GetShape();
  Vector3 dims_in_blocks = tiling_scheme.GetBlockCounts();
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  TF_ASSIGN_OR_RETURN(TilingThreadIdInfo thread_id_info,
                      EmitThreadIdInfo(builder, tiling_scheme, index_ty));

  KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);

  const llvm_ir::IrArray::Index block_coords(
      thread_id_info.block_id,
      ShapeUtil::MakeShape(PRED /*arbitrary*/, dims_in_blocks), builder);

  std::array<llvm::Value*, 3> tile_dimensions;
  for (int i = 0; i < 3; ++i) {
    int64_t block_tile_size = tiling_scheme.GetBlockTileSize()[i];
    if (dims_in_elems[i] % block_tile_size == 0) {
      // The block tile size evenly divides the tiled shape -> no need to emit
      // the bounds check.
      tile_dimensions[i] = constant(block_tile_size);
    } else {
      // Only the last tile in each dimension may not have full size.
      llvm::Value* is_last = builder->CreateICmpEQ(
          block_coords[i], constant(dims_in_blocks[i] - 1));
      int64_t partial_row =
          dims_in_elems[i] - (dims_in_blocks[i] - 1) * block_tile_size;
      tile_dimensions[i] =
          builder->CreateSelect(is_last, constant(partial_row),
                                constant(block_tile_size), "tile_bound");
    }
  }

  llvm_ir::IrArray::Index tile_origin = [&] {
    std::vector<llvm::Value*> elem_multi_index = block_coords.multidim();
    llvm::Type* index_ty = block_coords.GetType();
    for (int i = 0; i < TilingScheme::DimTot; ++i) {
      elem_multi_index[i] = builder->CreateMul(
          block_coords[i],
          llvm::ConstantInt::get(index_ty, tiling_scheme.GetBlockTileSize()[i]),
          "tile_origin." + std::to_string(i));
    }
    return llvm_ir::IrArray::Index(elem_multi_index, tiling_scheme.GetShape(),
                                   index_ty);
  }();

  tile_generator(thread_id_info, tile_origin, tile_dimensions);
  return {{tile_dimensions, tile_origin, thread_id_info}};
}

}  // namespace gpu
}  // namespace xla

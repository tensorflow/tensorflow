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
            tile_dimensions[dim], thread_id_info.strides[dim],
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

// Emits the LLVM values for thread_id, thread_id.x, thread_id.y and lane
// id.
//
// Returns a struct containing these values.
//
// In the presence of thread scaling in tiling scheme may return early if the
// combination of thread_id/block_id does not correspond to a real block.
// Assumes the current function returns void.
absl::StatusOr<TilingThreadIdInfo> EmitThreadIdInfo(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* index_ty) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  llvm::Value* thread_id_physical = EmitThreadId(
      builder, tiling_scheme.GetNumThreadsPerBlockPhysical(), index_ty);
  int64_t num_blocks = tiling_scheme.GetNumBlocksPhysical();
  if (num_blocks > (int64_t)std::numeric_limits<uint32_t>::max()) {
    return FailedPrecondition(
        "Number of physical blocks (%d) does not fit in an i32 in tiling "
        "scheme: %s",
        num_blocks, tiling_scheme.ToString());
  }
  llvm::Value* block_id_physical = EmitBlockId(builder, num_blocks, index_ty);

  // More than one thread in the z axis is currently not supported by the
  // index computation. Since the indexing is a bit complicated (with respect to
  // strides and starts and "virtual scaling"), there's no obvious way to extend
  // it right now.
  CHECK_EQ(tiling_scheme.GetThreadsPerBlock()[TilingScheme::DimZ], 1);

  llvm::Value* thread_id_logical = builder->CreateURem(
      thread_id_physical, constant(tiling_scheme.GetNumThreadsPerBlock()));
  llvm::Value* scaling = builder->CreateUDiv(
      thread_id_physical, constant(tiling_scheme.GetNumThreadsPerBlock()));
  llvm::Value* block_id_logical = builder->CreateAdd(
      builder->CreateMul(block_id_physical,
                         constant(tiling_scheme.GetThreadIdScalingFactor())),
      scaling);

  llvm::Value* num_threads_x_v =
      constant(tiling_scheme.GetThreadsPerBlock()[TilingScheme::DimX]);

  llvm::Value* block_exists = builder->CreateICmpULT(
      block_id_logical, constant(tiling_scheme.GetNumBlocks()));
  llvm_ir::EmitEarlyReturn(block_exists, builder);

  std::array<llvm::Value*, 3> thread_ids{
      constant(0),  // See above, there must be 1 thread in the z axis.
      builder->CreateUDiv(thread_id_logical, num_threads_x_v, "thread_id.y"),
      builder->CreateURem(thread_id_logical, num_threads_x_v, "thread_id.x")};
  std::array<llvm::Value*, 3> start_offsets{
      constant(0), thread_ids[TilingScheme::DimY],
      builder->CreateMul(thread_ids[TilingScheme::DimX],
                         constant(tiling_scheme.GetVectorSize()))};
  std::array<llvm::Value*, 3> strides{
      constant(1),
      constant(tiling_scheme.GetThreadsPerBlock()[TilingScheme::DimY]),
      constant(1)  // Not really, see EmitTileRec.
  };

  auto* lane_id =
      builder->CreateURem(thread_id_logical, constant(WarpSize()), "lane_id");
  return TilingThreadIdInfo{
      thread_id_logical, thread_ids,       start_offsets, strides,
      lane_id,           block_id_logical, scaling};
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

llvm_ir::IrArray::Index GetUnnormalizedIndex(
    const llvm_ir::IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* builder,
    absl::Span<const int64_t> dims_in_elems) {
  CHECK_EQ(normalized_shape_index.size(), 3);
  // If the normalization only add a new dimensions of size 1,
  // generate simpler indexing. LLVM doesn't always simplify the more
  // complicated indexing and this prevents it from vectorizing some
  // cases. We do this only for major_to_minor memory layout.
  if (unnormalized_shape.rank() == 2 && unnormalized_shape.has_layout() &&
      unnormalized_shape.dimensions()[0] == normalized_shape_index.dims()[1] &&
      unnormalized_shape.dimensions()[1] == normalized_shape_index.dims()[2] &&
      unnormalized_shape.layout().minor_to_major(1) == 0) {
    CHECK_EQ(normalized_shape_index.dims()[0], 1);
    auto multidim = normalized_shape_index.multidim();
    return llvm_ir::IrArray::Index({multidim[1], multidim[2]},
                                   unnormalized_shape,
                                   normalized_shape_index.GetType());
  }
  if (unnormalized_shape.rank() == 2 && unnormalized_shape.has_layout() &&
      unnormalized_shape.dimensions()[0] == normalized_shape_index.dims()[2] &&
      unnormalized_shape.dimensions()[1] == normalized_shape_index.dims()[1] &&
      unnormalized_shape.layout().minor_to_major(1) == 1) {
    CHECK_EQ(normalized_shape_index.dims()[0], 1);
    auto multidim = normalized_shape_index.multidim();
    return llvm_ir::IrArray::Index({multidim[2], multidim[1]},
                                   unnormalized_shape,
                                   normalized_shape_index.GetType());
  }
  return normalized_shape_index.SourceIndexOfBitcast(
      ShapeUtil::MakeShape(F32, dims_in_elems), unnormalized_shape, builder);
}

}  // namespace gpu
}  // namespace xla

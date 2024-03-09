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

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
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
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

void EmitTileRec(const TilingThreadIdInfo& thread_id_info, const Tiling& tiling,
                 int dim, absl::InlinedVector<llvm::Value*, 4> tile_idx,
                 absl::Span<llvm::Value* const> tile_dimensions,
                 llvm::IRBuilder<>* b, const TileElementGenerator& emit_elem) {
  llvm::Type* index_ty = thread_id_info.thread_id->getType();
  auto constant = [&](int64_t val) {
    return llvm::ConstantInt::get(index_ty, val);
  };

  auto recurse = [&] {
    if (dim == tile_idx.size() - 1) {
      emit_elem(tile_idx);
    } else {
      EmitTileRec(thread_id_info, tiling, dim + 1, tile_idx, tile_dimensions, b,
                  emit_elem);
    }
  };

  bool unroll = tiling.GetLoopsToUnroll()[dim];
  KernelSupportLibrary ksl(b, unroll ? llvm_ir::UnrollMode::kFullyUnroll
                                     : llvm_ir::UnrollMode::kDefaultUnroll);

  if (tiling.GetBlockTileSize()[dim] == 1) {
    tile_idx[dim] = constant(0);
    recurse();
  } else if (unroll) {
    // TODO(jreiffers): Check if this unrolling does anything useful.
    int64_t stride = tiling.GetThreadsPerBlock()[dim];
    int64_t dim_size = tiling.GetThreadTileSize()[dim];

    auto make_loop = [&](bool emit_bounds_checks) {
      auto body = [&, emit_bounds_checks](llvm::Value* i) {
        tile_idx[dim] = b->CreateAdd(i, thread_id_info.thread_ids[dim]);
        if (emit_bounds_checks) {
          auto* in_bounds =
              b->CreateICmpULT(tile_idx[dim], tile_dimensions[dim]);
          ksl.If("x_in_tile", in_bounds, recurse);
        } else {
          recurse();
        }
      };
      return [&, body] {
        ksl.For(absl::StrCat("loop", dim), constant(0),
                constant(dim_size * stride), constant(stride), body);
      };
    };
    if (stride > 1 && dim_size > 1) {
      // Most tiles will be full, so we emit a single bounds check for those.
      auto* is_full_tile = b->CreateICmpEQ(
          constant(tiling.GetBlockTileSize()[dim]), tile_dimensions[dim]);
      ksl.If("is_full_tile", is_full_tile, make_loop(false), make_loop(true));
    } else {
      make_loop(true)();
    }
  } else {
    // All dimensions are strided (thread 0 processes elements 0, num_threads,
    // num_threads+2, ...; thread 1 processes elements 1, num_threads + 1 and so
    // on).
    ksl.For(absl::StrCat("loop", dim), /*start=*/thread_id_info.thread_ids[dim],
            /*end=*/tile_dimensions[dim],
            /*step=*/tiling.GetThreadsPerBlock()[dim], [&](llvm::Value* i) {
              tile_idx[dim] = i;
              recurse();
            });
  }
}

}  // namespace

void EmitTile(llvm::IRBuilder<>* builder, const Tiling& tiling,
              const TilingThreadIdInfo& thread_id_info,
              absl::Span<llvm::Value* const> tile_dimensions,
              const TileElementGenerator& emit_elem_function) {
  absl::InlinedVector<llvm::Value*, 4> tile_idx(tiling.GetShape().size());
  EmitTileRec(thread_id_info, tiling, 0, tile_idx, tile_dimensions, builder,
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
                              llvm::cast<llvm::Instruction>(block_id),
                              builder->GetInsertBlock()->getModule());
  }
  auto ret = builder->CreateIntCast(block_id, index_ty, /*isSigned=*/true);
  ret->setName("block.id.x");
  return ret;
}

// Emits current thread id with the given type.
//
// Sets the return value range to [0, threads_per_block).
llvm::Value* EmitThreadId(llvm::IRBuilder<>* builder, int64_t threads_per_block,
                          llvm::Type* index_ty) {
  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::CallInst* thread_id =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, builder);
  llvm_ir::AddRangeMetadata(0, threads_per_block, thread_id,
                            builder->GetInsertBlock()->getModule());
  auto ret = builder->CreateIntCast(thread_id, index_ty, /*isSigned=*/true);
  ret->setName("thread.id.x");
  return ret;
}

// Emits the LLVM values for thread_id, block_id, coordinates of the current
// tile and strides of the loops to iterate over the current tile.
absl::StatusOr<TilingThreadIdInfo> EmitThreadIdInfo(llvm::IRBuilder<>* builder,
                                                    const Tiling& tiling,
                                                    llvm::Type* index_ty) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  int64_t num_blocks = tiling.GetNumBlocks();
  if (num_blocks > (int64_t)std::numeric_limits<uint32_t>::max()) {
    return FailedPrecondition(
        "Number of physical blocks (%d) does not fit in an i32 in tiling "
        "scheme: %s",
        num_blocks, tiling.ToString());
  }

  TilingThreadIdInfo info;
  info.thread_id =
      EmitThreadId(builder, tiling.GetNumThreadsPerBlock(), index_ty);
  info.block_id = EmitBlockId(builder, num_blocks, index_ty);

  for (auto [dim, stride] : llvm::enumerate(tiling.GetThreadStrides())) {
    int64_t size = tiling.GetThreadsPerBlock()[dim];
    if (size == 1) {
      info.thread_ids.emplace_back(constant(0));
    } else {
      auto& dim_id = info.thread_ids.emplace_back(info.thread_id);
      if (stride > 1) {
        dim_id = builder->CreateUDiv(dim_id, constant(stride));
      }
      if (dim) {
        dim_id = builder->CreateURem(dim_id, constant(size));
      }
      dim_id->setName(absl::StrCat("thread.id.", dim));
    }
  }

  info.lane_id =
      builder->CreateURem(info.thread_id, constant(WarpSize()), "lane_id");
  return info;
}

}  // namespace

absl::StatusOr<TilingKernelInfo> EmitTilingKernel(
    llvm::IRBuilder<>* builder, const Tiling& tiling, llvm::Type* index_ty,
    const TileGenerator& tile_generator) {
  absl::Span<const int64_t> dims_in_elems = tiling.GetShape();
  const auto& block_counts = tiling.GetBlockCounts();
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  TF_ASSIGN_OR_RETURN(TilingThreadIdInfo thread_id_info,
                      EmitThreadIdInfo(builder, tiling, index_ty));

  KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);

  const llvm_ir::IrArray::Index block_coords(
      thread_id_info.block_id,
      ShapeUtil::MakeShape(PRED /*arbitrary*/, block_counts), builder);

  absl::InlinedVector<llvm::Value*, 4> tile_dimensions;
  for (int i = 0; i < block_counts.size(); ++i) {
    int64_t block_tile_size = tiling.GetBlockTileSize()[i];
    if (dims_in_elems[i] % block_tile_size == 0) {
      // The block tile size evenly divides the tiled shape -> no need to emit
      // the bounds check.
      tile_dimensions.push_back(constant(block_tile_size));
    } else {
      // Only the last tile in each dimension may not have full size.
      llvm::Value* is_last =
          builder->CreateICmpEQ(block_coords[i], constant(block_counts[i] - 1));
      int64_t partial_row =
          dims_in_elems[i] - (block_counts[i] - 1) * block_tile_size;
      tile_dimensions.push_back(builder->CreateSelect(
          is_last, constant(partial_row), constant(block_tile_size),
          absl::StrCat("tile_bound.", i)));
    }
  }

  llvm_ir::IrArray::Index tile_offset = [&] {
    std::vector<llvm::Value*> elem_multi_index = block_coords.multidim();
    llvm::Type* index_ty = block_coords.GetType();
    for (int i = 0; i < block_counts.size(); ++i) {
      elem_multi_index[i] = builder->CreateMul(
          block_coords[i],
          llvm::ConstantInt::get(index_ty, tiling.GetBlockTileSize()[i]),
          absl::StrCat("tile_origin.", i));
    }
    return llvm_ir::IrArray::Index(elem_multi_index, tiling.GetShape(),
                                   index_ty);
  }();

  tile_generator(thread_id_info, tile_offset, tile_dimensions);
  return {{tile_dimensions, tile_offset, thread_id_info}};
}

}  // namespace gpu
}  // namespace xla

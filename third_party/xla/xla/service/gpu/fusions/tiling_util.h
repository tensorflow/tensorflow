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
#ifndef XLA_SERVICE_GPU_FUSIONS_TILING_UTIL_H_
#define XLA_SERVICE_GPU_FUSIONS_TILING_UTIL_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Describes tiling used by the kernel.
//
// Used by reduction and transpose emitters.
class Tiling {
 public:
  Tiling(absl::Span<const int64_t> shape, absl::Span<const int64_t> tile_sizes,
         absl::Span<const int64_t> num_threads,
         // By default, don't unroll anything.
         absl::InlinedVector<bool, 4> loops_to_unroll = {})
      : shape_{shape.begin(), shape.end()},
        tile_sizes_per_thread_{tile_sizes.begin(), tile_sizes.end()},
        tile_sizes_per_block_(shape.size()),
        num_threads_{num_threads.begin(), num_threads.end()},
        num_blocks_(shape.size()),
        loops_to_unroll_(loops_to_unroll) {
    for (int64_t i = 0; i < shape.size(); ++i) {
      tile_sizes_per_block_[i] = tile_sizes[i] * num_threads[i];
      CHECK_NE(tile_sizes_per_block_[i], 0);
      num_blocks_[i] = CeilOfRatio(shape[i], tile_sizes_per_block_[i]);
      CHECK_NE(num_blocks_[i], 0);
    }
    if (loops_to_unroll_.empty()) loops_to_unroll_.resize(shape.size());
  }

  std::string ToString() const {
    return absl::StrJoin(
        {absl::StrFormat("shape = {%s}", absl::StrJoin(shape_, ", ")),
         absl::StrFormat("tile_sizes = {%s}",
                         absl::StrJoin(tile_sizes_per_thread_, ", ")),
         absl::StrFormat("num_threads = {%s}",
                         absl::StrJoin(num_threads_, ", "))},
        ", ");
  }

  // Number of elements in each dimension.
  const absl::InlinedVector<int64_t, 4>& GetShape() const { return shape_; }
  xla::Shape GetXlaShape(PrimitiveType element_type = F32) const {
    return ShapeUtil::MakeShape(element_type, shape_);
  }

  const absl::InlinedVector<int64_t, 4>& GetBlockCounts() const {
    return num_blocks_;
  }

  // Tile size for each thread.
  //
  // Equals to the number of iterations in the loop each tile will make.
  const absl::InlinedVector<int64_t, 4>& GetThreadTileSize() const {
    return tile_sizes_per_thread_;
  }

  // Tile size for an entire thread block.
  const absl::InlinedVector<int64_t, 4>& GetBlockTileSize() const {
    return tile_sizes_per_block_;
  }

  const absl::InlinedVector<int64_t, 4>& GetThreadsPerBlock() const {
    return num_threads_;
  }

  // Returns the strides of the thread index dimensions wrt. the linear thread
  // id.
  absl::InlinedVector<int64_t, 4> GetThreadStrides() const {
    return *ShapeUtil::ByteStrides(ShapeUtil::MakeShape(U8, num_threads_));
  }

  // Returns the strides of the block index dimensions wrt. the linear block id.
  absl::InlinedVector<int64_t, 4> GetBlockStrides() const {
    return *ShapeUtil::ByteStrides(ShapeUtil::MakeShape(U8, num_blocks_));
  }

  int64_t GetNumThreadsPerBlock() const { return Product(num_threads_); }

  int64_t GetNumBlocks() const { return Product(num_blocks_); }

  const absl::InlinedVector<bool, 4>& GetLoopsToUnroll() const {
    return loops_to_unroll_;
  }

 private:
  // The number of elements in each dimension.
  absl::InlinedVector<int64_t, 4> shape_;

  // The number of elements for each dimension of a tile.
  absl::InlinedVector<int64_t, 4> tile_sizes_per_thread_;
  absl::InlinedVector<int64_t, 4> tile_sizes_per_block_;

  absl::InlinedVector<int64_t, 4> num_threads_;
  absl::InlinedVector<int64_t, 4> num_blocks_;

  absl::InlinedVector<bool, 4> loops_to_unroll_;
};

struct TilingThreadIdInfo {
  llvm::Value* thread_id;

  absl::InlinedVector<llvm::Value*, 4> thread_ids;

  // Lane id: `thread_id % WarpSize`
  llvm::Value* lane_id;

  // Block id.
  llvm::Value* block_id;
};

struct TilingKernelInfo {
  // Tiling bounds.
  absl::InlinedVector<llvm::Value*, 4> output_tile_bounds;

  // Starting tile, as calculated from block id only.
  llvm_ir::IrArray::Index tile_origin;

  // Thread meta-info.
  TilingThreadIdInfo thread_id_info;
};

// A function to generate the code to emit the entire tile.
//
// index: Absolute coordinate of the start of the tile in input.
// tile_dimensions: Size of the tile
using TileGenerator =
    std::function<void(const TilingThreadIdInfo& thread_id_info,
                       const llvm_ir::IrArray::Index& tile_start_index,
                       absl::Span<llvm::Value* const> tile_dimensions)>;

// A function object to generate code to process one element in a tile.
//
// index_in_tile: the current coordinates within the tile. To get the global
// coordinates, use `tile_start_index.AddOffset(index_in_tile, ...)`.
using TileElementGenerator =
    std::function<void(absl::Span<llvm::Value* const> index_in_tile)>;

// Emits code to iterate through a tile with given tile dimensions and generate
// elements using the callback.
void EmitTile(llvm::IRBuilder<>* builder, const Tiling& tiling,
              const TilingThreadIdInfo& thread_id_info,
              absl::Span<llvm::Value* const> tile_dimensions,
              const TileElementGenerator& emit_elem_function);

// Emits a kernel for the hlo instruction using the given kernel mapping
// scheme.
absl::StatusOr<TilingKernelInfo> EmitTilingKernel(
    llvm::IRBuilder<>* builder, const Tiling& tiling, llvm::Type* index_ty,
    const TileGenerator& tile_element_generator);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TILING_UTIL_H_

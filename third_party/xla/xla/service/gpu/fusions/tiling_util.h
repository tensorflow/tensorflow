/*Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Describes tiling used by the kernel.
//
// Used by reductions and 021 transpose algorithm. Both algorithms operate over
// "logical" 3D views over input arrays, hence tiling and number of threads
// information has only 3 dimensions.
//
// In the presence of virtual threadIdx/blockIdx scaling, all accessors are
// "logical", unless otherwise specified.
class TilingScheme {
 public:
  enum { DimZ = 0, DimY, DimX, DimTot };

  enum IndexingOrder {
    // Thread reads consecutive elements.
    LinearIndexingX,
    // Thread reads strided elements while keeping memory coalescing.
    StridedIndexingX,
  };

  TilingScheme(Vector3 dims_in_elems, Vector3 tile_sizes, Vector3 num_threads,
               IndexingOrder indexing_order, int vector_size,
               int scaling_factor, Vector2 tiling_dimensions = Vector2{1, 2})
      : dims_in_elems_(dims_in_elems),
        tile_sizes_per_thread_(tile_sizes),
        tile_sizes_per_block_{num_threads[0] * tile_sizes[0],
                              num_threads[1] * tile_sizes[1],
                              num_threads[2] * tile_sizes[2]},
        tiling_dimensions_(tiling_dimensions),
        num_threads_(num_threads),
        indexing_order_(indexing_order),
        vector_size_(vector_size),
        thread_id_virtual_scaling_(scaling_factor) {
    CHECK_EQ(tile_sizes[2] % vector_size_, 0)
        << "tile sizes = " << absl::StrJoin(tile_sizes, ", ")
        << "; vector size = " << vector_size_;
  }

  static std::string IndexingOrderToString(IndexingOrder order) {
    switch (order) {
      case LinearIndexingX:
        return "linear";
      case StridedIndexingX:
        return "strided";
    }
  }

  std::string ToString() const {
    return absl::StrJoin(
        {absl::StrFormat("dims_in_elems = {%s}",
                         absl::StrJoin(dims_in_elems_, ", ")),
         absl::StrFormat("tile_sizes = {%s}",
                         absl::StrJoin(tile_sizes_per_thread_, ", ")),
         absl::StrFormat("num_threads = {%s}",
                         absl::StrJoin(num_threads_, ", ")),
         absl::StrFormat("indexing_order = %s",
                         IndexingOrderToString(indexing_order_)),
         absl::StrFormat("vector_size = %d", vector_size_),
         absl::StrFormat("thread_id_virtual_scaling = %d",
                         thread_id_virtual_scaling_),
         absl::StrFormat("tiling_dimensions = {%s}",
                         absl::StrJoin(tiling_dimensions_, ", "))},
        ", ");
  }

  // Number of elements in each dimension (Z/Y/X respectively).
  const Vector3& GetShape() const { return dims_in_elems_; }

  Vector3 GetBlockCounts() const {
    return {GetBlockCount(0), GetBlockCount(1), GetBlockCount(2)};
  }

  // Tile size for each thread.
  //
  // Equals to the number of iterations in the loop each tile will make.
  const Vector3& GetThreadTileSize() const { return tile_sizes_per_thread_; }

  // Tile size for an entire thread block.
  const Vector3& GetBlockTileSize() const { return tile_sizes_per_block_; }

  // The tiling dimension for dimension 'd' of the shared memory tile.
  int64_t GetTilingDimension(int d) const { return tiling_dimensions_.at(d); }

  // Number of logical threads per block.
  const Vector3& GetThreadsPerBlock() const { return num_threads_; }
  int64_t GetNumThreadsPerBlock() const {
    return num_threads_[0] * num_threads_[1] * num_threads_[2];
  }

  // Number of logical blocks.
  int64_t GetNumBlocks() const {
    auto counts = GetBlockCounts();
    return counts[0] * counts[1] * counts[2];
  }

  // Number of physical blocks launched (with scaling applied).
  int64_t GetNumBlocksPhysical() const {
    return CeilOfRatio(GetNumBlocks(), thread_id_virtual_scaling_);
  }

  // Number of physical threads per block launched (with scaling applied).
  int64_t GetNumThreadsPerBlockPhysical() const {
    return num_threads_[0] * num_threads_[1] * num_threads_[2] *
           thread_id_virtual_scaling_;
  }

  IndexingOrder GetIndexingOrder() const { return indexing_order_; }
  int GetVectorSize() const { return vector_size_; }

  // Scaling factor for transforming physical threadId to logical.
  int GetThreadIdScalingFactor() const { return thread_id_virtual_scaling_; }

 private:
  // Number of blocks required to "cover" the given dimension.
  int64_t GetBlockCount(int d) const {
    return CeilOfRatio(dims_in_elems_[d], tile_sizes_per_block_[d]);
  }

  // The number of elements in each dimension.
  Vector3 dims_in_elems_;

  // The number of elements for each dimension of a tile.
  Vector3 tile_sizes_per_thread_;
  Vector3 tile_sizes_per_block_;

  // The dimensions which are used for the shared memory tile.
  Vector2 tiling_dimensions_;

  // Number of threads implicitly assigned to each dimension.
  Vector3 num_threads_;

  IndexingOrder indexing_order_;

  // Vector size for dimension X.
  int vector_size_;

  // Scaling apply to transform physical threadIdx into logical.
  int64_t thread_id_virtual_scaling_ = 1;
};

// Contains threading information. Note that for performance we might apply
// thread id "scaling" where the physical thread id (to achieve good SM
// occupancy) will differ from logical thread id. This struct contains
// logical thread ids, along with meta-information about the scaling applied.
struct TilingThreadIdInfo {
  TilingThreadIdInfo(llvm::Value* thread_id, llvm::Value* thread_id_x,
                     llvm::Value* thread_id_y, llvm::Value* lane_id,
                     llvm::Value* block_id, llvm::Value* scaling)
      : thread_id(thread_id),
        thread_id_x(thread_id_x),
        thread_id_y(thread_id_y),
        lane_id(lane_id),
        block_id(block_id),
        scaling(scaling) {}

  llvm::Value* thread_id;

  // X-coordinate calculated from thread id: `thread_id % num_threads_x`
  llvm::Value* thread_id_x;

  // Y-coordinate calculated from thread id: `thread_id / num_threads_x`
  llvm::Value* thread_id_y;

  // Lane id: `thread_id % WarpSize`
  llvm::Value* lane_id;

  // Block id.
  llvm::Value* block_id;

  // Emits GEP into a shared memory, taking virtual thread scaling into
  // account. Automatically inserts the first zero required by LLVM GEP.
  // Defined on ThreadIdInfo to keep `scaling` private.
  //
  // Same semantics as CreateInBoundsGEP.
  llvm::Value* GEPIntoSharedMemory(
      llvm::IRBuilder<>* b, llvm::GlobalVariable* shared,
      absl::Span<llvm::Value* const> idx_major_to_minor,
      const llvm::Twine& name = "") const;

  // Calculate the pointee type of the llvm::Value returned by
  // GEPIntoSharedMemory
  llvm::Type* GEPIntoSharedMemoryType(
      llvm::GlobalVariable* shared,
      absl::Span<llvm::Value* const> idx_major_to_minor) const;

 private:
  llvm::Value* scaling;
};

struct TilingKernelInfo {
  // Tiling bounds.
  std::array<llvm::Value*, 2> output_tile_bounds;

  // Starting tile, as calculated from block id only.
  llvm_ir::IrArray::Index tile_origin;

  // Thread meta-info.
  TilingThreadIdInfo thread_id_info;
};

// A function to generate the code to emit the entire tile.
//
// index: Absolute coordinate of the start of the tile in input.
// tile_dimensions: Size of the tile
using TileElementGenerator =
    std::function<void(const TilingThreadIdInfo& thread_id_info,
                       const llvm_ir::IrArray::Index& index,
                       std::array<llvm::Value*, 2> tile_dimensions)>;

// A function object to generate code to process one element in a tile.
//
// index: the index for the first output element of the current thread.
// y_loc: The y coordinate within a tile.
// x_loc: The x coordinate within a tile.
using EmitTileElementFunction =
    std::function<void(const TilingThreadIdInfo& thread_id_info,
                       const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
                       llvm::Value* x_loc)>;

// Emits code to iterate through a 2-dimensional tile with a given tile
// dimensions and given strides, and call the callback at each iteration.,
//
// thread_id_y` and `thread_id_x` are the intra-tile coordinates for
// the first element to process, and `index` is the index for the origin of
// the tile. Emits bounds check to ensure that each processed element
// is within the boundary defined by `tile_dimensions`.
//
// Rough pseudocode:
//
// Given: tile_dimensions, x_offset, y_offset
//
// for (y = 0; y < tile_dimensions[0]; y += num_threads_y) {
//   for (x = 0; x < tile_dimensions[1]; x++) {
//
//     y_pos = y_offset + y
//     x_pos = x_offset + x * stride
//
//     if (x_loc < tile_width) {
//       emit_elem_function(y_offset + y, x_loc);
//     }
//   }
// }
//
void EmitTile(llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
              const llvm_ir::IrArray::Index& tile_origin_index,
              const TilingThreadIdInfo& thread_id_info,
              std::array<llvm::Value*, 2> tile_dimensions,
              const EmitTileElementFunction& emit_elem_function);

// Emits a kernel for the hlo instruction using the given kernel mapping
// scheme.
absl::StatusOr<TilingKernelInfo> EmitTilingKernel(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* index_ty, const TileElementGenerator& tile_element_generator);

llvm_ir::IrArray::Index GetUnnormalizedIndex(
    const llvm_ir::IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* builder,
    absl::Span<const int64_t> dims_in_elems);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TILING_UTIL_H_

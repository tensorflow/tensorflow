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

#include <functional>

#include "xla/service/gpu/kernel_mapping_scheme.h"
#include "xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace gpu {

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
StatusOr<TilingKernelInfo> EmitTilingKernel(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* index_ty, const TileElementGenerator& tile_element_generator);

llvm_ir::IrArray::Index GetUnnormalizedIndex(
    const llvm_ir::IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* builder,
    absl::Span<const int64_t> dims_in_elems);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TILING_UTIL_H_

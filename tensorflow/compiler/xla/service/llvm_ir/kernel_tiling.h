/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_TILING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_TILING_H_

#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace llvm_ir {

// About 0-2-1 transpose:
//
// If a shape can be viewed as three logical components 0-1-2 in the order of
// major to minor, a 0-2-1-transpose changes the order of such logical
// components to 0-2-1. We call the shape being transposed the input shape and
// the transposed shape the output shape. The logical view of the input/output
// shapes for the transpose are called the 0-1-2/0-2-1 shapes or the normalized
// shapes. The original input/output shapes are called unnormalized shapes.
//
// If `b` is a 0-2-1 transpose of `a` in 0-1-2, return the dimensions for the
// normalized shape of `b` or the 0-2-1 shape.
absl::optional<std::vector<int64> > FindTranspose021(const Shape& a,
                                                     const Shape& b);

// A tile is a spatial subdivision of a tensor. We group tensor elements into
// tiles so that we can launch kernels to process the tensor elements in blocks
// of tiles.
//
// A kernel mapping scheme describes a method to partition the tensors accessed
// by an unnested HLO instruction into tiles and blocks of tiles, and the
// associated information to use hardware threads to process the tensor elements
// in blocks of tiles.
//
// Currently, there are two main use cases for a tiling scheme. First, we
// implement kernels with 0-2-1 memory transpose using shared memory to improve
// memory access pattern. Second, we implement reduction to contiguous
// dimensions in layout, with or without memory tranpsose, to achieve better
// memory access pattern as well as to reduce the need numbers of executed
// expensive instructions, such as thread synchronization related instructions
// and atomic operations. For both use cases, we can apply a normalization to
// the original tensors, to collapse contiguous dimensions for the same purpose
// and produce normlized three dimensional tensors. For this reason, the tiling
// scheme class only needs to handle normalized three dimensional tensors and
// two dimensional tiles.
//
// The current implementation of the class is somewhat NVIDIA GPU oriented. This
// situation can be improved when there is a need though. The idea of 0-2-1
// transpose using shared memory can be found in the following CUDA algorithm in
// TensorFlow: https://goo.gl/MStRV6.
//
// We use a thread block to process a tile because we want to use the HW thread
// block synchronization primitives to synchronize the processing of all the
// elements in the same tile. A thread block can be viewed as a two dimensional
// array of threads, described by the number of threads for the Y and X
// dimensions. A thread block (num_threads_y, num_threads_x) processes a tile of
// (tile_size_y, tile_size_x) as follows: each thread in the thread block
// processes one element in the tile so that all the threads in the thread block
// together process a subdivision of the tile that has the same dimension as the
// thread block array. Then the thread block moves on to process the next
// subdivision of the tile until the whole tile is processed. Therefore, each
// thread in the thread block processes
// tile_size_x/num_threads_x * tile_size_y/num_threads_y elements in a tile.
//
// There are situations where we want a thread block to process multiple
// tiles. We can't group those tiles into a bigger tiles because we limit a tile
// to a two dimensional spatial subdivision of a tensor. For example, when we
// use tiling to implement reduction with tranpose, we want the partial sum
// produced by each thread to accumulate values for more elements before using
// shlf_down and atomic_add instructions for further reduction, to amortize the
// cost of such expensive instructions. The concept of tile block is introduced
// for this purpose. A tile block is a three dimensional array of tiles, of
// which some dimensions may be degenerated to only one tile.
class KernelMappingScheme {
 public:
  enum { DimZ = 0, DimY, DimX, DimTot };

 public:
  // dims_in_elems: the normalized tensor dimensions.
  // req_block_sizes: the requested block size in number of tiles for each
  //   dimension. The actual block size is set to min(req_block_size,
  //   dims_in_number_of_blocks).
  explicit KernelMappingScheme(absl::Span<const int64> dims_in_elems,
                               int64 tile_size_y, int64 tile_size_x,
                               absl::Span<const int64> req_block_sizes,
                               int64 num_threads_y, int64 num_threads_x,
                               llvm::IRBuilder<>* b);

  absl::Span<const int64> GetDimensionsInElements() const {
    return dims_in_elems_;
  }
  absl::Span<const int64> GetDimensionsInTiles() const {
    return dims_in_tiles_;
  }
  absl::Span<const int64> GetDimensionsInBlocks() const {
    return dims_in_blocks_;
  }

  int64 GetNumberOfTilesInTotal() const {
    return absl::c_accumulate(dims_in_tiles_, 1LL, std::multiplies<int64>());
  }
  int64 GetNumberOfTilesInOneBlock() const {
    return absl::c_accumulate(block_sizes_, 1, std::multiplies<int64>());
  }

  int64 GetNumberOfBlocks() const {
    return absl::c_accumulate(dims_in_blocks_, 1, std::multiplies<int64>());
  }

  int64 GetTileSizeForDimension(int d) const {
    DCHECK(d >= DimZ && d <= DimX);
    return tile_sizes_[d];
  }
  int64 GetTileSizeForDimensionX() const {
    return GetTileSizeForDimension(DimX);
  }
  int64 GetTileSizeForDimensionY() const {
    return GetTileSizeForDimension(DimY);
  }

  absl::Span<const int64> GetBlockSizes() const { return block_sizes_; }

  int64 GetNumberOfThreadsForDimensionX() const { return num_threads_x_; }
  int64 GetNumberOfThreadsForDimensionY() const { return num_threads_y_; }

  int64 GetThreadsPerTile() const {
    return GetNumberOfThreadsForDimensionX() *
           GetNumberOfThreadsForDimensionY();
  }

  IrArray::Index EmitBlockIndex(llvm::Type* index_ty);
  // Returns the index for the first tile in the block with the given block
  // index.
  IrArray::Index GetTileIndexForBlockOrigin(const IrArray::Index& block_index);
  // Returns the index for the first element in the tile with the given tile
  // index.
  IrArray::Index GetElementIndexForTileOrigin(const IrArray::Index& tile_index);

  std::tuple<llvm::Value*, llvm::Value*> EmitThreadYXCoordinate(
      llvm::Type* index_ty);

  IrArray::Index GetUnnormalizedIndex(
      const IrArray::Index& normalized_shape_index,
      const Shape& unnormalized_shape);

  llvm::GlobalVariable* GetSharedMemoryBufferForElementType(
      llvm::Type* elem_ty, absl::string_view buffer_name);

 private:
  llvm::IRBuilder<>* b_;
  // The number of elements in each dimension.
  absl::Span<const int64> dims_in_elems_;

  // The number of elements for each dimension of a tile.
  std::vector<int64> tile_sizes_;
  // The number of tiles in each dimension. It is computed from dims_in_elem_
  // and tile_sizes_.
  std::vector<int64> dims_in_tiles_;

  // The number of tiles for each dimension of a tile block.
  std::vector<int64> block_sizes_;
  // The number of blocks in each dimension of a tile block. It is computed from
  // dims_in_tile_ and block_sizes_.
  std::vector<int64> dims_in_blocks_;

  // Number of threads used to process elements in the X direction of a tile.
  int64 num_threads_x_;
  // Number of threads used to process elements in the Y direction of a tile.
  int64 num_threads_y_;
};

// A class to represent information for tiled parameters to support IR emission
// for 021 transpose.
class TiledParameterInfo {
 public:
  TiledParameterInfo(absl::Span<llvm::Value* const> param_buffers,
                     llvm::Value* y, llvm::Value* x)
      : param_buffers_(param_buffers), y_(y), x_(x) {}

  llvm::Value* x() const { return x_; }
  llvm::Value* y() const { return y_; }

  void set_x(llvm::Value* x) { x_ = x; }
  void set_y(llvm::Value* y) { y_ = y; }

  llvm::Value* GetBufferForParameter(int64 index) const {
    return param_buffers_[index];
  }

 private:
  // Param_buffers_[i] stores the tile buffer for the ith parameter or nullptr
  // if the parameter is not tiled.
  absl::Span<llvm::Value* const> param_buffers_;
  // The y coordinate within a tile.
  llvm::Value* y_;
  // The x coordinate within a tile.
  llvm::Value* x_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_TILING_H_

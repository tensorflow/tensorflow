/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace gpu {

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
// dimensions in layout, with or without memory transpose, to achieve better
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
  enum IndexingOrder {
    // Thread reads consecutive elements.
    LinearIndexingX,
    // Thread reads strided elements while keeping memory coalescing.
    StridedIndexingX,
    // Thread reads a few consecutive elements then take a strided
    // step. This can trigger vectorized reads and keep memory
    // coalescing.
    StridedLinearIndexingX
  };

  KernelMappingScheme(absl::Span<const int64> dims_in_elems,
                      absl::Span<const int64> tile_sizes, int64_t num_threads_y,
                      int64_t num_threads_x, IndexingOrder indexing_order,
                      int vector_size, bool is_row_contiguous = false)
      : dims_in_elems_{dims_in_elems[0], dims_in_elems[1], dims_in_elems[2]},
        tile_sizes_{tile_sizes[0], tile_sizes[1], tile_sizes[2]},
        num_threads_x_(num_threads_x),
        num_threads_y_(num_threads_y),
        indexing_order_(indexing_order),
        vector_size_(vector_size),
        is_row_contiguous_(is_row_contiguous) {
    CHECK_EQ(tile_sizes[1] % num_threads_y_, 0);
    CHECK_EQ(tile_sizes[2] % num_threads_x_, 0);
    VLOG(10) << "dims_in_elems_ = " << absl::StrJoin(dims_in_elems_, ",");
    if (indexing_order != LinearIndexingX) {
      // StridedIndexingX, and StridedLinearIndexingX
      // is for the purpose of vectorization, which requires
      // GetTileSizeFor(DimX) to be a multiplier of num_threads_x_.
      CHECK_EQ(GetTileSizeFor(DimX) % num_threads_x_, 0);
    }
  }

  // Number of elements in each dimension (Z/Y/X respectively).
  absl::Span<const int64> GetDimsInElems() const { return dims_in_elems_; }

  int64 GetNumberOfBlocks() const {
    return CeilOfRatio(dims_in_elems_[0], GetTileSizeZ()) *
           CeilOfRatio(dims_in_elems_[1], GetTileSizeY()) *
           CeilOfRatio(dims_in_elems_[2], GetTileSizeX());
  }

  // Tile size for a given dimensions. Tiles are assigned per thread block,
  // and are processed by all threads in the block.
  int64 GetTileSizeFor(int d) const { return tile_sizes_.at(d); }

  int64 GetTileSizeZ() const { return GetTileSizeFor(DimZ); }
  int64 GetTileSizeX() const { return GetTileSizeFor(DimX); }
  int64 GetTileSizeY() const { return GetTileSizeFor(DimY); }

  int64 GetNumThreadsX() const { return num_threads_x_; }
  int64 GetNumThreadsY() const { return num_threads_y_; }

  int64 GetThreadsPerBlock() const {
    return GetNumThreadsX() * GetNumThreadsY();
  }

  IndexingOrder GetIndexingOrder() const { return indexing_order_; }
  int GetVectorSize() const { return vector_size_; }
  bool GetRowContiguous() const { return is_row_contiguous_; }

 private:
  // The number of elements in each dimension.
  const std::array<int64, 3> dims_in_elems_;

  // The number of elements for each dimension of a tile.
  const std::array<int64, 3> tile_sizes_;

  // Number of threads used to process elements in the X direction of a tile.
  const int64 num_threads_x_;

  // Number of threads used to process elements in the Y direction of a tile.
  const int64 num_threads_y_;

  // When num_threads_x threads process a total of tile_size_x
  // elements in the X dimension of a tile, each threads process
  // n=tile_size_x/num_threads_x elements.
  // indexing_order defines which tile's elements each thread reads.
  const IndexingOrder indexing_order_;

  // vector_size_ only supported for row reduction and must be a divisor
  // of tile_sizes_[2]/num_threads_x.  Interesting values are 2 and 4
  // to trigger vectorized loads on GPUs while keeping memory
  // coalescing.
  const int vector_size_;
  const bool is_row_contiguous_;
};

// Information to support the code generation for a tiled reduction kernel.
using AddressVector = absl::InlinedVector<llvm::AllocaInst*, 1>;
class ReductionCodegenInfo {
 public:
  explicit ReductionCodegenInfo(KernelMappingScheme mapping_scheme,
                                int num_partial_results, bool is_row_reduction)
      : mapping_scheme_(mapping_scheme),
        num_partial_results_(num_partial_results),
        is_row_reduction_(is_row_reduction) {
    if (num_partial_results > 1) {
      CHECK_EQ(num_partial_results, (mapping_scheme.GetTileSizeX() /
                                     mapping_scheme.GetNumThreadsX()));
    }
  }

  const KernelMappingScheme& GetKernelMappingScheme() const {
    return mapping_scheme_;
  }

  // Gets writeable pointer to the address (or addresses) used to store
  // reduction accumulators.
  AddressVector* GetMutablePartialResultAddresses() {
    return &partial_result_addresses_;
  }

  // Returns the address (addresses) of the reduction accumulators.
  absl::Span<llvm::AllocaInst* const> GetPartialResultAddresses() const {
    return partial_result_addresses_;
  }

  // Mutable pointer to the address of the input element to perform the
  // reduction with.
  AddressVector* GetMutableReductionInputAddresses() {
    return &reduction_input_addresses_;
  }

  std::vector<llvm::Value*>* GetMutableInitialValues() {
    return &initial_values_;
  }

  absl::Span<llvm::Value* const> GetInitialValues() const {
    return initial_values_;
  }

  // Returns the address of the input element to perform the reduction with.
  absl::Span<llvm::AllocaInst* const> GetReductionInputAddresses() const {
    return reduction_input_addresses_;
  }

  int GetNumPartialResults() const { return num_partial_results_; }
  bool IsRowReduction() const { return is_row_reduction_; }

  // Gets a pointer to a mutable shared cache used by reduction.
  std::vector<llvm::GlobalVariable*>* GetMutableSharedCache() {
    return &shared_cache_;
  }

  // Shared cache used for reduction.
  absl::Span<llvm::GlobalVariable* const> GetSharedCache() const {
    return shared_cache_;
  }

 private:
  std::vector<llvm::GlobalVariable*> shared_cache_;
  std::vector<llvm::Value*> initial_values_;
  const KernelMappingScheme mapping_scheme_;
  AddressVector partial_result_addresses_;
  AddressVector reduction_input_addresses_;
  int num_partial_results_;
  bool is_row_reduction_;
};

}  // end namespace gpu
}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_

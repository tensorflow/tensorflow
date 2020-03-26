/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_

#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"

namespace ruy {

enum class BlockMapTraversalOrder {
  // Plain old row-by-row or column-by-column traversal.
  kLinear,
  // Fractal Z-order curve, https://en.wikipedia.org/wiki/Z-order_curve
  kFractalZ,
  // Variant of Z-order doing a U instead of a Z.
  kFractalU,
  // Hilbert curve, https://en.wikipedia.org/wiki/Hilbert_curve
  kFractalHilbert
};

// A BlockMap describes a tiling of a matrix, typically the destination matrix
// of a matrix multiplication computation. As is standard in matrix
// multiplication, a tile is called a "block".
//
// Ruy subdivides work by blocks of the destination matrix: each thread fully
// computes a block at once, then moves on to another block; each block is
// produced by a single thread.
//
// This ensures that the workloads for each block are mutually independent,
// which reduces synchronization requirements.
//
// Typically, a matrix multiplication will early on create a BlockMap by
// calling MakeBlockMap. It will then query the number of blocks in that
// BlockMap by calling NumBlocks. It will then create a single atomic integer
// counter indexing these blocks, called the 'index', and will distribute
// work to its N threads by ensuring that each thread works on disjoint sets
// of index values. For a given index value, the thread will call
// GetBlockByIndex to get the corresponding block, then GetBlockMatrixCoords
// to find the actual row and column numbers of this block.
//
// There are two nested levels of subdivision. On a local level, the matrix is
// tiled into a square NxN grid where N is a power of two, specifically:
//   N = 2^num_blocks_base_log2.
//
// At a larger scale, around these blocks, there may be one further
// level of subdivision, in only one dimension: either along rows or along
// columns. That is used to handle arbitrarily rectangular matrices. The
// aforementioned high-level block grid is square, so it does not readily fit
// well very rectangular matrices.
//
// Taking together these two nested levels of subdivision, the effective
// tiling is by
//   2^(num_blocks_base_log2 + rows_rectangularness_log2)
// blocks in the row dimension, and by
//   2^(num_blocks_base_log2 + cols_rectangularness_log2)
// blocks in the column dimension. See NumBlocksOfRows, NumBlocksOfCols.
//
// Either rows_rectangularness_log2 or cols_rectangularness_log2 must be zero.
//
// Finally, this BlockMap is designed to operate under alignment constraints:
// two fields, kernel_rows and kernel_cols, describe the requested alignment
// of the effective grid in both dimensions. The idea is to feed matrix
// multiplication kernels with tiles that fit their width as much as possible.
// Of course, if rows (resp. cols) is not a multiple of kernel_rows (resp.
// kernel_cols) then some tile will have to have unaligned size. BlockMap
// will only allow that to happen in the last position along each axis, so
// as to minimize the overhead incurred onto the matrix multiplication kernels.
struct BlockMap {
  // The number of threads to use (to distribute the blocks to).
  int thread_count;
  // The order in which to traverse the matrix of which this BlockMap represents
  // a tiling (hereafter "the matrix").
  BlockMapTraversalOrder traversal_order;
  // The dimensions of the block_map, that is, of the destination
  // matrix rounded up to next multiples of kernel_dims.
  SidePair<int> dims;
  // Log2 of the minimum number of subdivisions of the grid along either axis.
  int num_blocks_base_log2;
  // Log2 of the additional subdivision of the rows/columns axis.
  SidePair<int> rectangularness_log2;
  // Requested alignment of the subdivisions of the grid along the rows/columns
  // axis.
  SidePair<int> kernel_dims;
  // Internal helper. Minimum number of rows/columns in each block.
  SidePair<int> small_block_dims;
  // Internal helper. Number of blocks along each dimension that need to have
  // their size in that dimension be given by (small_block_dims + kernel_dims)
  // instead of just small_block_dims.
  SidePair<int> large_blocks;
};

// Returns the traversal order to be used for the given matrix multiplication
// parameters.
BlockMapTraversalOrder GetTraversalOrder(int rows, int cols, int depth,
                                         int lhs_scalar_size,
                                         int rhs_scalar_size,
                                         int local_data_cache_size,
                                         int shared_data_cache_size);

// Create a BlockMap suitable for tiling the destination matrix in a
// matrix multiplication with the given parameters.
void MakeBlockMap(int rows, int cols, int depth, int kernel_rows,
                  int kernel_cols, int lhs_scalar_size, int rhs_scalar_size,
                  int tentative_thread_count, Path path,
                  int local_data_cache_size, int shared_data_cache_size,
                  BlockMap* block_map);

// Maps an integer index to a block position in the grid.
void GetBlockByIndex(const BlockMap& block_map, int index,
                     SidePair<int>* block);

// Given a block position in the grid, returns its actual
// position in the matrix that the BlockMap refers to in the dimension
// referred to by `side`: along rows if side==kLhs, along columns if
// side==kRhs.
void GetBlockMatrixCoords(Side side, const BlockMap& block_map, int block,
                          int* start, int* end);

// Given a block position in the grid, returns its actual
// position in the matrix that the BlockMap refers to in terms of
// actual row/column indices.
void GetBlockMatrixCoords(const BlockMap& block_map, const SidePair<int>& block,
                          SidePair<int>* start, SidePair<int>* end);

// Returns the number of grid subdivisions along the rows dimension (if
// side == kLhs) or columns dimension (if side == kRhs).
inline int NumBlocksPerSide(Side side, const BlockMap& block_map) {
  return 1 << (block_map.num_blocks_base_log2 +
               block_map.rectangularness_log2[side]);
}

// Returns the overall number of blocks in
// the BlockMap. The valid index values to pass to GetBlockByIndex are the
// integers from 0 to N-1 where N is the value returned here.
//
// Note that it is always true that
//   NumBlocks == NumBlocksOfRows * NumBlocksOfCols
// because either rows_rectangularness_log2 or cols_rectangularness_log2 is 0.
inline int NumBlocks(const BlockMap& block_map) {
  return 1 << (2 * block_map.num_blocks_base_log2 +
               block_map.rectangularness_log2[Side::kLhs] +
               block_map.rectangularness_log2[Side::kRhs]);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_

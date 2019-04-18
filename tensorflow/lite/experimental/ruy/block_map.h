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

#include <cstdint>

namespace ruy {

// The value and even the meaning of this constant are empirically
// determined. Coarsely speaking, it's compared with the size of source
// LHS and RHS operands to determine whether they are big enough to be worth
// traversing in a more complicated "cache friendly" order. The current
// value is roughly the minimum size of a L1 cache on any CPU that we currently
// care about, e.g. ARM Cortex-A53. But we honestly don't even know the precise
// extent to which this should be related to L1 cache size.
//
// A lower value is not necessarily 'safer' from a cache-friendliness
// perspective: it means switching sooner (at smaller sizes) to more complicated
// traversal orders, which might be adversarial to the CPU's auto-prefetching
// or to the TLB.
static constexpr int kCacheFriendlyLoopThreshold = 32 * 1024;

enum class BlockMapTraversalOrder {
  // Plain old row-by-row or column-by-column traversal.
  kLinear,
  // Fractal Z-order curve, https://en.wikipedia.org/wiki/Z-order_curve
  kFractalZ,
  // Variant of Z-order doing a U instead of a Z.
  kFractalU
  // TODO(benoitjacob) add Hilbert curve order. More complex decoding might be
  // worth it.
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
// There are two nested levels of subdivision. On a high level, the matrix is
// tiled into a square NxN grid where N is a power of to, specifically:
//   N = 2^num_blocks_base_log2.
//
// At a smaller scale, within each of these blocks, there may be one further
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
  // The order in which to traverse the matrix of which this BlockMap represents
  // a tiling (hereafter "the matrix").
  BlockMapTraversalOrder traversal_order;
  // The number of rows in the matrix.
  int rows;
  // The number of columns in the matrix.
  int cols;
  // Log2 of the minimum number of subdivisions of the grid along either axis.
  int num_blocks_base_log2;
  // Log2 of the additional subdivision of the rows axis.
  int rows_rectangularness_log2;
  // Log2 of the additional subdivision of the columns axis.
  int cols_rectangularness_log2;
  // Requested alignment of the subdivions grid along the rows axis.
  int kernel_rows;
  // Requested alignment of the subdivions grid along the columns axis.
  int kernel_cols;
  // Internal helper. Minimum number of rows in each block.
  std::uint16_t smallr;
  // Internal helper. Minimum number of columns in each block.
  std::uint16_t smallc;
  // Internal helper. Number of rows that would be missed at the end if
  // all blocks had exactly `smallr` rows.
  std::uint16_t missr;
  // Internal helper. Number of columns that would be missed at the end if
  // all blocks had exactly `smallc` columns.
  std::uint16_t missc;
};

// Create a BlockMap suitable for tiling the destination matrix in a
// matrix multiplication with the given parameters.
void MakeBlockMap(int rows, int cols, int depth, int kernel_rows,
                  int kernel_cols, int lhs_scalar_size, int rhs_scalar_size,
                  BlockMap* block_map);

// Maps an integer index to a (block_r, block_c) block position in the grid.
void GetBlockByIndex(const BlockMap& block_map, std::uint32_t index,
                     std::uint16_t* block_r, std::uint16_t* block_c);

// Given a (block_r, block_c) block position in the grid, returns its actual
// position in the matrix that the BlockMap refers to in terms of
// actual row/column indices: starting at row start_r and column start_c,
// ending at row (end_r - 1) and column (end_c - 1).
void GetBlockMatrixCoords(const BlockMap& block_map, std::uint16_t block_r,
                          std::uint16_t block_c, int* start_r, int* start_c,
                          int* end_r, int* end_c);

// Returns the number of grid subdivisions along the rows dimension.
inline std::uint16_t NumBlocksOfRows(const BlockMap& block_map) {
  return 1 << (block_map.num_blocks_base_log2 +
               block_map.rows_rectangularness_log2);
}

// Returns the number of grid subdivisions along the columns dimension.
inline std::uint16_t NumBlocksOfCols(const BlockMap& block_map) {
  return 1 << (block_map.num_blocks_base_log2 +
               block_map.cols_rectangularness_log2);
}

// Returns the overall number of blocks in
// the BlockMap. The valid index values to pass to GetBlockByIndex are the
// integers from 0 to N-1 where N is the value returned here.
//
// Note that it is always true that
//   NumBlocks == NumBlocksOfRows * NumBlocksOfCols
// because either rows_rectangularness_log2 or cols_rectangularness_log2 is 0.
inline std::uint32_t NumBlocks(const BlockMap& block_map) {
  return 1 << (2 * block_map.num_blocks_base_log2 +
               block_map.rows_rectangularness_log2 +
               block_map.cols_rectangularness_log2);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_

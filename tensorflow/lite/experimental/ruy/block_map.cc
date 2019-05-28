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

#include "tensorflow/lite/experimental/ruy/block_map.h"

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"

namespace ruy {

void GetBlockByIndex(const BlockMap& block_map, std::uint32_t index,
                     std::uint16_t* block_r, std::uint16_t* block_c) {
  gemmlowp::ScopedProfilingLabel label("GetBlockByIndex");
  std::uint16_t rectr =
      index & ((1 << block_map.rows_rectangularness_log2) - 1);
  std::uint16_t rectc =
      index & ((1 << block_map.cols_rectangularness_log2) - 1);

  std::uint16_t n1 = index >> (block_map.rows_rectangularness_log2 +
                               block_map.cols_rectangularness_log2);
  RUY_DCHECK_EQ(index, (n1 << (block_map.rows_rectangularness_log2 +
                               block_map.cols_rectangularness_log2)) +
                           rectr + rectc);

  std::uint16_t br, bc;
  if (block_map.traversal_order == BlockMapTraversalOrder::kLinear) {
    br = n1 & ((1 << block_map.num_blocks_base_log2) - 1);
    bc = n1 >> block_map.num_blocks_base_log2;
  } else {
    // Decode fractal z-order
    std::uint16_t n2 =
        (n1 & 0x9999) | ((n1 & 0x4444) >> 1) | ((n1 & 0x2222) << 1);
    std::uint16_t n4 =
        (n2 & 0xc3c3) | ((n2 & 0x3030) >> 2) | ((n2 & 0x0c0c) << 2);
    std::uint16_t n8 =
        (n4 & 0xf00f) | ((n4 & 0x0f00) >> 4) | ((n4 & 0x00f0) << 4);
    br = n8 & 0xff;
    bc = n8 >> 8;
    if (block_map.traversal_order == BlockMapTraversalOrder::kFractalU) {
      // Change fractal z-order to u-order
      br ^= bc;
    }
  }

  br = (br << block_map.rows_rectangularness_log2) + rectr;
  bc = (bc << block_map.cols_rectangularness_log2) + rectc;

  // Store
  *block_r = br;
  *block_c = bc;
}

namespace {

int floor_log2_quotient(int num, int denom) {
  if (num <= denom) {
    return 0;
  }
  int log2_quotient = floor_log2(num) - ceil_log2(denom);
  if ((denom << (log2_quotient + 1)) <= num) {
    log2_quotient++;
  }
  return log2_quotient;
}

}  // namespace

void MakeBlockMap(int rows, int cols, int depth, int kernel_rows,
                  int kernel_cols, int lhs_scalar_size, int rhs_scalar_size,
                  BlockMap* block_map) {
  gemmlowp::ScopedProfilingLabel label("MakeBlockMap");
  RUY_DCHECK_GE(rows, kernel_rows);
  RUY_DCHECK_GE(cols, kernel_cols);

  block_map->traversal_order = BlockMapTraversalOrder::kLinear;
  if ((RUY_OPT_SET & RUY_OPT_FRACTAL) &&
      (rows * lhs_scalar_size + cols * rhs_scalar_size) * depth >=
          kCacheFriendlyLoopThreshold) {
    block_map->traversal_order = (RUY_OPT_SET & RUY_OPT_FRACTAL_U)
                                     ? BlockMapTraversalOrder::kFractalU
                                     : BlockMapTraversalOrder::kFractalZ;
  }

  int rows_rectangularness_log2 = 0;
  int cols_rectangularness_log2 = 0;
  if (rows >= cols) {
    rows_rectangularness_log2 = floor_log2_quotient(rows, cols);
    RUY_DCHECK_GE(rows >> rows_rectangularness_log2, cols);
    RUY_DCHECK_EQ(cols_rectangularness_log2, 0);
  }
  if (cols >= rows) {
    cols_rectangularness_log2 = floor_log2_quotient(cols, rows);
    RUY_DCHECK_GE(cols >> cols_rectangularness_log2, rows);
    RUY_DCHECK_EQ(rows_rectangularness_log2, 0);
  }

  RUY_DCHECK(!rows_rectangularness_log2 || !cols_rectangularness_log2);

  const int size = std::min(rows, cols);
  const int size_floor_log2 = floor_log2(size);
  const int depth_ceil_log2 = ceil_log2(depth);
  const int kernel_width_log2 = ceil_log2(std::max(kernel_cols, kernel_rows));

  // l1_size_log2 was originally, coarsely speaking the number of rows of LHS,
  // or the number of columns of RHS in a matrix multiplication that we expect,
  // to fit in L1 cache.
  //
  // This initial rationale is not necessarily still relevant. The logic below
  // was determined empirically, not in a principled way.
  int l1_size_log2;
  if (size_floor_log2 <= 3) {
    l1_size_log2 = size_floor_log2;
  } else if (size_floor_log2 <= 6) {
    l1_size_log2 = 4;
  } else {
    l1_size_log2 = 5;
  }

  // The 15 here implicitly encodes target a 32k L1 cache (2^15 == 32k).
  // Once again this only has a distant memory of being originally motivated
  // by such clear principles linking this logic to cache sizes.
  l1_size_log2 = std::min(
      l1_size_log2, 15 - depth_ceil_log2 -
                        ceil_log2(std::max(lhs_scalar_size, rhs_scalar_size)));
  l1_size_log2 = std::max(l1_size_log2, kernel_width_log2);
  l1_size_log2 = std::min(l1_size_log2, size_floor_log2);
  l1_size_log2 = std::max(l1_size_log2, size_floor_log2 - 8);

  int num_blocks_base_log2 = size_floor_log2 - l1_size_log2;
  RUY_DCHECK_GE(num_blocks_base_log2, 0);
  RUY_DCHECK_LE(num_blocks_base_log2, 8);
  if (num_blocks_base_log2 == 0) {
    if ((rows % kernel_rows) || (cols % kernel_cols)) {
      num_blocks_base_log2 = 1;
    }
  }
  RUY_DCHECK_LE(num_blocks_base_log2 + rows_rectangularness_log2, 16);
  RUY_DCHECK_LE(num_blocks_base_log2 + cols_rectangularness_log2, 16);

  int rows_rounded_up = round_up_pot(rows, kernel_rows);
  int cols_rounded_up = round_up_pot(cols, kernel_cols);

  const int num_blocks_of_rows_log2 =
      num_blocks_base_log2 + rows_rectangularness_log2;
  const int num_blocks_of_cols_log2 =
      num_blocks_base_log2 + cols_rectangularness_log2;

  std::uint16_t smallr =
      round_down_pot(rows_rounded_up >> num_blocks_of_rows_log2, kernel_rows);
  std::uint16_t smallc =
      round_down_pot(cols_rounded_up >> num_blocks_of_cols_log2, kernel_cols);
  std::uint16_t missr =
      round_up_pot(rows_rounded_up - (smallr << num_blocks_of_rows_log2),
                   kernel_rows) /
      kernel_rows;
  std::uint16_t missc =
      round_up_pot(cols_rounded_up - (smallc << num_blocks_of_cols_log2),
                   kernel_cols) /
      kernel_cols;

  block_map->rows = rows;
  block_map->cols = cols;
  block_map->kernel_rows = kernel_rows;
  block_map->kernel_cols = kernel_cols;
  block_map->num_blocks_base_log2 = num_blocks_base_log2;
  block_map->rows_rectangularness_log2 = rows_rectangularness_log2;
  block_map->cols_rectangularness_log2 = cols_rectangularness_log2;
  block_map->smallr = smallr;
  block_map->smallc = smallc;
  block_map->missr = missr;
  block_map->missc = missc;
}

void GetBlockMatrixCoords(const BlockMap& block_map, std::uint16_t block_r,
                          std::uint16_t block_c, int* start_r, int* start_c,
                          int* end_r, int* end_c) {
  gemmlowp::ScopedProfilingLabel label("GetBlockMatrixCoords");
  int sr = block_r * block_map.smallr +
           std::min(block_r, block_map.missr) * block_map.kernel_rows;
  int er = sr + block_map.smallr +
           (block_r < block_map.missr) * block_map.kernel_rows;
  int sc = block_c * block_map.smallc +
           std::min(block_c, block_map.missc) * block_map.kernel_cols;
  int ec = sc + block_map.smallc +
           (block_c < block_map.missc) * block_map.kernel_cols;
  sc = round_down_pot(sc, block_map.kernel_cols);
  ec = round_down_pot(ec, block_map.kernel_cols);
  sr = round_down_pot(sr, block_map.kernel_rows);
  er = round_down_pot(er, block_map.kernel_rows);

  ec = std::min(ec, block_map.cols);
  er = std::min(er, block_map.rows);
  sc = std::max(0, ec - round_up_pot(ec - sc, block_map.kernel_cols));
  sr = std::max(0, er - round_up_pot(er - sr, block_map.kernel_rows));

  *start_c = sc;
  *end_c = ec;
  *start_r = sr;
  *end_r = er;

  RUY_DCHECK_LE(ec, block_map.cols);
  RUY_DCHECK_LE(er, block_map.rows);
  RUY_DCHECK_LT(sc, ec);
  RUY_DCHECK_LT(sr, er);
  RUY_DCHECK_GE(sc, 0);
  RUY_DCHECK_GE(sr, 0);
}

}  // namespace ruy

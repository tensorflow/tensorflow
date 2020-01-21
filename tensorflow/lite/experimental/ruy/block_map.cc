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

#include <algorithm>
#include <cstdint>

#ifdef RUY_MAKEBLOCKMAP_DEBUG
#include <cstdio>
#include <cstdlib>
#include <string>
#endif

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"

namespace ruy {

void GetBlockByIndex(const BlockMap& block_map, int index,
                     SidePair<int>* block) {
  profiler::ScopeLabel label("GetBlockByIndex");
  const std::uint32_t index_u32 = index;

  const std::uint32_t num_blocks_per_local_curve =
      1u << (2 * block_map.num_blocks_base_log2);
  const std::uint32_t n1 = index_u32 & (num_blocks_per_local_curve - 1);

  SidePair<int> local_pos;
  if (block_map.traversal_order == BlockMapTraversalOrder::kLinear) {
    local_pos[Side::kLhs] = n1 & ((1u << block_map.num_blocks_base_log2) - 1);
    local_pos[Side::kRhs] = n1 >> block_map.num_blocks_base_log2;
  } else {
    // Decode fractal z-order
    const std::uint32_t n2 = (n1 & 0x99999999u) | ((n1 & 0x44444444u) >> 1) |
                             ((n1 & 0x22222222u) << 1);
    const std::uint32_t n4 = (n2 & 0xc3c3c3c3u) | ((n2 & 0x30303030u) >> 2) |
                             ((n2 & 0x0c0c0c0cu) << 2);
    const std::uint32_t n8 = (n4 & 0xf00ff00fu) | ((n4 & 0x0f000f00u) >> 4) |
                             ((n4 & 0x00f000f0u) << 4);
    const std::uint32_t n16 = (n8 & 0xff0000ffu) | ((n8 & 0x00ff0000u) >> 8) |
                              ((n8 & 0x0000ff00u) << 8);
    local_pos[Side::kLhs] = n16 & 0xffff;
    local_pos[Side::kRhs] = n16 >> 16;
    if (block_map.traversal_order == BlockMapTraversalOrder::kFractalU) {
      // Change fractal z-order to u-order
      local_pos[Side::kLhs] ^= local_pos[Side::kRhs];
    }
  }

  const std::uint32_t rectangular_index =
      index_u32 >> 2 * block_map.num_blocks_base_log2;
  for (Side side : {Side::kLhs, Side::kRhs}) {
    const std::uint32_t mask = (1u << block_map.rectangularness_log2[side]) - 1;
    const int rectangular_offset = (rectangular_index & mask)
                                   << block_map.num_blocks_base_log2;
    (*block)[side] = local_pos[side] + rectangular_offset;
  }
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

BlockMapTraversalOrder GetTraversalOrder(
    int rows, int cols, int depth, int lhs_scalar_size, int rhs_scalar_size,
    int cache_friendly_traversal_threshold) {
  if (RUY_OPT_ENABLED(RUY_OPT_FRACTAL) &&
      (rows * lhs_scalar_size + cols * rhs_scalar_size) * depth >=
          cache_friendly_traversal_threshold) {
    return RUY_OPT_ENABLED(RUY_OPT_FRACTAL_U)
               ? BlockMapTraversalOrder::kFractalU
               : BlockMapTraversalOrder::kFractalZ;  // NOLINT
    // (clang-tidy complains that the 'else' here is never taken).
  } else {
    return BlockMapTraversalOrder::kLinear;
  }
}

// Computes the rectangularness of the matrix shape (rows, cols). This is
// essentially just the log2 of the quotient (rows / cols). The kernel_rows and
// kernel_cols only get into the picture for clamping bounds but don't affect
// the generic computation.
void GetRectangularness(int rows, int cols, int kernel_rows, int kernel_cols,
                        int* rows_rectangularness_log2,
                        int* cols_rectangularness_log2) {
  *rows_rectangularness_log2 = 0;
  *cols_rectangularness_log2 = 0;

  // In GEMV-ish cases, that is when kernel blocks are as narrow as the kernel
  // itself, we risk having too small kernel blocks for good kernel
  // amortization. We avoid that by limiting recangularness so that kernel
  // blocks are not too tiny at least in that dimension. Specifically, we try to
  // have at least (2^min_kernel_inner_loop_runs_log2) kernels fitting in each
  // kernel block along the large dimension.
  const int min_kernel_inner_loop_runs_log2 = 3;
  if (rows > cols) {
    int cols_of_kernel_inner_loop_runs_log2 =
        ceil_log2(cols) - pot_log2(kernel_cols);
    int min_rows_of_kernel_inner_loop_runs_log2 =
        std::max(0, min_kernel_inner_loop_runs_log2 -
                        cols_of_kernel_inner_loop_runs_log2);
    *rows_rectangularness_log2 =
        std::min(floor_log2_quotient(rows, cols),
                 std::max(0, floor_log2(rows) - pot_log2(kernel_rows) -
                                 min_rows_of_kernel_inner_loop_runs_log2));
    // Sanity check that we did not over-estimate rows_rectangularness_log2.
    RUY_DCHECK_GE(rows >> *rows_rectangularness_log2, cols);
  } else if (cols > rows) {
    int rows_of_kernel_inner_loop_runs_log2 =
        ceil_log2(rows) - pot_log2(kernel_rows);
    int min_cols_of_kernel_inner_loop_runs_log2 =
        std::max(0, min_kernel_inner_loop_runs_log2 -
                        rows_of_kernel_inner_loop_runs_log2);
    *cols_rectangularness_log2 =
        std::min(floor_log2_quotient(cols, rows),
                 std::max(0, floor_log2(cols) - pot_log2(kernel_cols) -
                                 min_cols_of_kernel_inner_loop_runs_log2));
    // Sanity check that we did not over-estimate cols_rectangularness_log2.
    RUY_DCHECK_GE(cols >> *cols_rectangularness_log2, rows);
  }
  RUY_DCHECK(!*rows_rectangularness_log2 || !*cols_rectangularness_log2);
}

// Computes a 'multithreading score'. When multithreading, we need there to
// be at least as many tiles as there are threads, and hopefully
// substantially more than that, so we benefit from ruy's ability to
// dispatch fine-grained workloads to threads.
int GetMultithreadingScore(int block_size_log2, int rows, int cols,
                           int tentative_thread_count) {
  const int num_full_blocks_of_rows = rows >> block_size_log2;
  const int num_full_blocks_of_cols = cols >> block_size_log2;
  const int candidate_num_full_blocks_log2 = floor_log2(
      std::max(1, num_full_blocks_of_rows * num_full_blocks_of_cols));

  // The values here have been tuned on ARM Cortex-A55.
  // We expect this to have to be tuned differently for other CPUs.
  if (tentative_thread_count == 1) {
    return 0;
  } else {
    const int blocks_per_thread_log2 =
        candidate_num_full_blocks_log2 - ceil_log2(tentative_thread_count);
    if (blocks_per_thread_log2 < 0) {
      return -64;
    } else if (blocks_per_thread_log2 == 0) {
      return -16;
    } else if (blocks_per_thread_log2 == 1) {
      return -8;
    } else if (blocks_per_thread_log2 == 2) {
      return 0;
    } else if (blocks_per_thread_log2 == 3) {
      return 8;
    } else {
      return 16;
    }
  }
}

// Computes a 'cache locality score'. This is the familiar notion that
// local working sets should be small enough to fit in some local data
// cache, by which we mean that typically L1 and possibly L2 caches, being
// local to each CPU core, tend to perform better than typical last-level
// (e.g. L3) caches shared among all cores. Here we aim to fit in a fast,
// local cache.
int GetCacheLocalityScore(int block_size_log2, int rows, int cols, int depth,
                          int kernel_rows_log2, int kernel_cols_log2,
                          int lhs_scalar_size, int rhs_scalar_size, Path path) {
  // In the narrow case (e.g. matrix*vector), each byte of the big operand
  // matrix (either LHS or RHS) is traversed only once, so any notion of data
  // locality is irrelevant. Ignore the 'cache locality score' by forcing it to
  // be 0 in that case.
  if (rows <= (1 << kernel_rows_log2) || cols <= (1 << kernel_cols_log2)) {
    return 0;
  }
  const int block_rows = std::min(1 << block_size_log2, rows);
  const int block_cols = std::min(1 << block_size_log2, cols);
#if RUY_PLATFORM(ARM_64)
  const int kLocalDataCacheSizeLog2 = path == Path::kNeonDotprod ? 17 : 15;
#elif RUY_PLATFORM(ARM_32)
  const int kLocalDataCacheSizeLog2 = 14;
#elif RUY_PLATFORM(X86)
  const int kLocalDataCacheSizeLog2 = 17;
#else
  const int kLocalDataCacheSizeLog2 = 14;
#endif
  const int lhs_bytes_log2 =
      pot_log2(lhs_scalar_size) + ceil_log2(block_rows * depth);
  const int rhs_bytes_log2 =
      pot_log2(rhs_scalar_size) + ceil_log2(block_cols * depth);
  const int total_read_bytes_log2 =
      1 + std::max(lhs_bytes_log2, rhs_bytes_log2);
  const int nonlocality_log2 = total_read_bytes_log2 - kLocalDataCacheSizeLog2;
  // The values here have been tuned on ARM Cortex-A55.
  // We expect this to have to be tuned differently for other CPUs.
  if (nonlocality_log2 < -1) {
    return 64;
  } else if (nonlocality_log2 == -1) {
    return 56;
  } else if (nonlocality_log2 == 0) {
    return 48;
  } else if (nonlocality_log2 == 1) {
    return 32;
  } else if (nonlocality_log2 == 2) {
    return 0;
  } else {
    return -64;
  }
}

// Compute a 'kernel amortization score'. This is the notion that very small
// tiles result in more overhead outside of kernels, more complex memory
// access patterns and less benefits from ruy's fat kernels, so we reward
// larger blocks more than smaller ones.
int GetKernelAmortizationScore(int block_size_log2, int rows, int cols,
                               int kernel_rows_log2, int kernel_cols_log2) {
  const int block_rows = std::min(1 << block_size_log2, rows);
  const int block_cols = std::min(1 << block_size_log2, cols);
  const int kernels_per_block_log2 =
      floor_log2(block_rows * block_cols) - kernel_rows_log2 - kernel_cols_log2;
  RUY_DCHECK_GE(kernels_per_block_log2, 0);
  // The values here have been tuned on ARM Cortex-A55.
  // We expect this to have to be tuned differently for other CPUs.
  if (kernels_per_block_log2 == 0) {
    return 0;
  } else if (kernels_per_block_log2 == 1) {
    return 8;
  } else if (kernels_per_block_log2 == 2) {
    return 16;
  } else if (kernels_per_block_log2 == 3) {
    return 24;
  } else if (kernels_per_block_log2 == 4) {
    return 32;
  } else if (kernels_per_block_log2 == 5) {
    return 40;
  } else if (kernels_per_block_log2 == 6) {
    return 48;
  } else if (kernels_per_block_log2 == 7) {
    return 56;
  } else {
    return 64;
  }
}

}  // namespace

void MakeBlockMap(int rows, int cols, int depth, int kernel_rows,
                  int kernel_cols, int lhs_scalar_size, int rhs_scalar_size,
                  int tentative_thread_count, Path path,
                  int cache_friendly_traversal_threshold, BlockMap* block_map) {
  profiler::ScopeLabel label("MakeBlockMap");

#ifdef RUY_MAKEBLOCKMAP_DEBUG
#if RUY_MAKEBLOCKMAP_DEBUG >= 2
  static constexpr bool debug_everytime = true;
#else
  static constexpr bool debug_everytime = false;
#endif
  static bool firsttime = true;
  if (firsttime || debug_everytime) {
    fprintf(stderr,
            "MakeBlockMap(rows=%d, cols=%d, depth=%d, kernel_rows=%d, "
            "kernel_cols=%d, lhs_scalar_size=%d, rhs_scalar_size=%d, "
            "tentative_thread_count=%d)\n",
            rows, cols, depth, kernel_rows, kernel_cols, lhs_scalar_size,
            rhs_scalar_size, tentative_thread_count);
  }
#endif

  RUY_DCHECK_GE(rows, kernel_rows);
  RUY_DCHECK_GE(cols, kernel_cols);
  RUY_DCHECK_EQ(rows % kernel_rows, 0);
  RUY_DCHECK_EQ(cols % kernel_cols, 0);

  block_map->traversal_order =
      GetTraversalOrder(rows, cols, depth, lhs_scalar_size, rhs_scalar_size,
                        cache_friendly_traversal_threshold);

  int rows_rectangularness_log2 = 0;
  int cols_rectangularness_log2 = 0;
  GetRectangularness(rows, cols, kernel_rows, kernel_cols,
                     &rows_rectangularness_log2, &cols_rectangularness_log2);

  const int kernel_rows_log2 = pot_log2(kernel_rows);
  const int kernel_cols_log2 = pot_log2(kernel_cols);
  const int kernel_size_log2 = std::max(kernel_cols_log2, kernel_rows_log2);

  const int size = std::min(rows, cols);
  const int size_log2 = std::max(kernel_size_log2, floor_log2(size));

  RUY_DCHECK_GE(size_log2, kernel_size_log2);

  // We are going to try candidate values for block_size_log2 ranging from
  // kernel_size_log2 to (kernel_size_log2 + kMaxKernelsPerBlockLog2).
  // For each of them we will compute a 'score' by adding individual scores
  // for a few different considerations, all of which is entirely empirical.
  // The values (and possibly the logic) around here are all subject to tuning
  // based on benchmarks on different hardware. The current values are based
  // on benchmarking on Qualcomm S855 (big and little cores), arm64,
  // kNeonDotprod, 8bit quantized path. Don't read too much into it, go ahead
  // and tune this as needed to achieve good performance elsewhere. Use
  // the unit test, block_map_test, to encode values that should be preserved
  // on specific architectures. Use RUY_MAKEBLOCKMAP_DEBUG to help tuning this.
  static constexpr int kMaxKernelsPerBlockLog2 = 6;
  const int max_block_size_log2 =
      std::min(size_log2, kernel_size_log2 + kMaxKernelsPerBlockLog2);
  int best_score = std::numeric_limits<int>::min();
  int best_score_block_size_log2 = -1;
  for (int block_size_log2 = kernel_size_log2;
       block_size_log2 <= max_block_size_log2; block_size_log2++) {
    const int multithreading_score = GetMultithreadingScore(
        block_size_log2, rows, cols, tentative_thread_count);
    const int cache_locality_score = GetCacheLocalityScore(
        block_size_log2, rows, cols, depth, kernel_rows_log2, kernel_cols_log2,
        lhs_scalar_size, rhs_scalar_size, path);
    const int kernel_amortization_score = GetKernelAmortizationScore(
        block_size_log2, rows, cols, kernel_rows_log2, kernel_cols_log2);
    const int score =
        multithreading_score + cache_locality_score + kernel_amortization_score;
#ifdef RUY_MAKEBLOCKMAP_DEBUG
    if (firsttime || debug_everytime) {
      fprintf(stderr,
              "block_size_log2=%d: score=%d multithreading_score=%d "
              "cache_locality_score=%d kernel_amortization_score=%d\n",
              block_size_log2, score, multithreading_score,
              cache_locality_score, kernel_amortization_score);
    }
#endif
    if (score >= best_score) {
      best_score = score;
      best_score_block_size_log2 = block_size_log2;
    }
  }

#ifdef RUY_MAKEBLOCKMAP_DEBUG
  if (firsttime || debug_everytime) {
    fprintf(stderr, "best_score_block_size_log2=%d\n",
            best_score_block_size_log2);
  }

  static const char* explicit_block_size_log2_env =
      getenv("RUY_MAKEBLOCKMAP_EXPLICIT_BLOCK_SIZE_LOG2");
  if (explicit_block_size_log2_env) {
    best_score_block_size_log2 = std::stoi(explicit_block_size_log2_env);
    if (firsttime || debug_everytime) {
      fprintf(stderr, "Overridden best_score_block_size_log2=%d\n",
              best_score_block_size_log2);
    }
  }
  firsttime = false;
#endif

  int num_blocks_base_log2 = size_log2 - best_score_block_size_log2;
  RUY_DCHECK_GE(num_blocks_base_log2, 0);

  const int num_blocks_of_rows_log2 =
      num_blocks_base_log2 + rows_rectangularness_log2;
  const int num_blocks_of_cols_log2 =
      num_blocks_base_log2 + cols_rectangularness_log2;

  const int smallr =
      round_down_pot(rows >> num_blocks_of_rows_log2, kernel_rows);
  const int smallc =
      round_down_pot(cols >> num_blocks_of_cols_log2, kernel_cols);
  const int missr =
      round_up_pot(rows - (smallr << num_blocks_of_rows_log2), kernel_rows) >>
      pot_log2(kernel_rows);
  const int missc =
      round_up_pot(cols - (smallc << num_blocks_of_cols_log2), kernel_cols) >>
      pot_log2(kernel_cols);

  block_map->dims[Side::kLhs] = rows;
  block_map->dims[Side::kRhs] = cols;
  block_map->kernel_dims[Side::kLhs] = kernel_rows;
  block_map->kernel_dims[Side::kRhs] = kernel_cols;
  block_map->num_blocks_base_log2 = num_blocks_base_log2;
  block_map->rectangularness_log2[Side::kLhs] = rows_rectangularness_log2;
  block_map->rectangularness_log2[Side::kRhs] = cols_rectangularness_log2;
  block_map->small_block_dims[Side::kLhs] = smallr;
  block_map->small_block_dims[Side::kRhs] = smallc;
  block_map->large_blocks[Side::kLhs] = missr;
  block_map->large_blocks[Side::kRhs] = missc;
  // Done last: NumBlocks needs some of the block_map fields to be already set.
  block_map->thread_count =
      std::min(tentative_thread_count, NumBlocks(*block_map));
}

void GetBlockMatrixCoords(Side side, const BlockMap& block_map, int block,
                          int* start, int* end) {
  profiler::ScopeLabel label("GetBlockMatrixCoords");
  *start = block * block_map.small_block_dims[side] +
           std::min(block, block_map.large_blocks[side]) *
               block_map.kernel_dims[side];
  *end =
      *start + block_map.small_block_dims[side] +
      (block < block_map.large_blocks[side] ? block_map.kernel_dims[side] : 0);

  RUY_DCHECK_EQ(0, *start % block_map.kernel_dims[side]);
  RUY_DCHECK_EQ(0, *end % block_map.kernel_dims[side]);
  RUY_DCHECK_LE(*end, block_map.dims[side]);
  RUY_DCHECK_LT(*start, *end);
  RUY_DCHECK_GE(*start, 0);
}

void GetBlockMatrixCoords(const BlockMap& block_map, const SidePair<int>& block,
                          SidePair<int>* start, SidePair<int>* end) {
  for (Side side : {Side::kLhs, Side::kRhs}) {
    GetBlockMatrixCoords(side, block_map, block[side], &(*start)[side],
                         &(*end)[side]);
  }
}

}  // namespace ruy

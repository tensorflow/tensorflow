/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/raft_select_k_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

constexpr int64_t kScratchAlignment = 256;
constexpr int64_t kMinSafetyMarginBytes = 4096;
constexpr int64_t kSafetyMarginDivisor = 16;
constexpr int64_t kFloatBytes = sizeof(float);
constexpr int64_t kIndexBytes = sizeof(uint32_t);
constexpr int64_t kWarpSize = 32;
constexpr int64_t kMaxThreadsPerSm = 2048;
constexpr int64_t kMaxWarpsPerSm = kMaxThreadsPerSm / kWarpSize;
constexpr int64_t kMaxBlocksPerSm = 32;

// Adds `max(4 KB, estimated_bytes / 16)` rounded up to 256-byte alignment to
// absorb initial base-pointer alignment padding in
// `FixedBufferDeviceMemoryResource` and minor workspace descriptor/counter
// growth across CCCL/CUB and RAFT updates within XLA GPU's AOT compatibility
// window.
int64_t AddSafetyMargin(int64_t estimated_bytes) {
  const int64_t margin = std::max<int64_t>(
      kMinSafetyMarginBytes, estimated_bytes / kSafetyMarginDivisor);
  return RoundUpTo(estimated_bytes + margin, kScratchAlignment);
}

// Mirrors `raft::matrix::detail::select::radix::impl::calc_buf_len<float,
// uint32_t, RatioT>(len)` in select_radix.cuh:
//   constexpr RatioT ratio = 2 + sizeof(uint32_t) * 2 / sizeof(float);  // 4
//   uint32_t buf_len = len / (ratio * 8);
//   return Pow2<64>::roundDown(buf_len);
// `radix_topk_one_block` instantiates `RatioT = unsigned`, whereas multi-block
// `radix_topk` uses the default `RatioT = float`, where IEEE-754 float
// promotion of `len > 2^24` can round up across a 64-element boundary.
template <typename RatioT>
int64_t CalcRadixBufLenFloat(uint32_t len) {
  constexpr RatioT kRatio = 2 + sizeof(uint32_t) * 2 / sizeof(float);
  const uint32_t buf_len = static_cast<uint32_t>(len / (kRatio * 8));
  return static_cast<int64_t>(buf_len & ~uint32_t{63});
}

// Bounds temporary workspace for RAFT select_warpsort:
//   tmp_val: num_of_block * batch * k * sizeof(float)
//   tmp_idx: num_of_block * batch * k * sizeof(uint32_t)
// WarpSort outputs are already sorted, so no Phase 2 sort runs.
//
// In raft/matrix/detail/select_warpsort.cuh (`block_kernel` has
// `__launch_bounds__(256)`), `calc_launch_parameter` chooses a multiple-of-32
// block size `32 * w` (`1 <= w <= 8`) and `min_grid_size <= blocks_per_sm *
// sm_count`, where `blocks_per_sm <= min(32, 64 / w)`. Because `num_of_block`
// for a fixed `w` is monotonically non-decreasing with `min_grid_size`, taking
// the maximum across `w in [1, 8]` upper-bounds `num_of_block` across all GPU
// architectures and compiler occupancy outcomes.
int64_t EstimateWarpSortScratchBytesFloat(int64_t batch, int64_t n, int64_t k,
                                          int64_t sm_count) {
  const int64_t capacity =
      static_cast<int64_t>(absl::bit_ceil(static_cast<uint32_t>(k)));
  const int64_t capacity_per_full_warp = std::max<int64_t>(capacity, kWarpSize);
  constexpr int64_t kMultiBlockLenFactor = 2;

  int64_t max_num_blocks = 1;
  for (int64_t num_warps = 1; num_warps <= 8; ++num_warps) {
    const int64_t max_blocks_per_sm_for_warps =
        std::min<int64_t>(kMaxBlocksPerSm, kMaxWarpsPerSm / num_warps);
    const int64_t max_min_grid_size = max_blocks_per_sm_for_warps * sm_count;
    if (batch >= max_min_grid_size) {
      continue;
    }
    const int64_t init_blocks = max_min_grid_size / batch;
    const int64_t init_len_per_block = CeilOfRatio(n, init_blocks);
    int64_t len_per_warp =
        RoundUpTo(CeilOfRatio(init_len_per_block, num_warps), kWarpSize);
    len_per_warp = std::max<int64_t>(
        len_per_warp, capacity_per_full_warp * kMultiBlockLenFactor);
    const int64_t len_per_block =
        std::min<int64_t>(n, num_warps * len_per_warp);
    const int64_t num_blocks = CeilOfRatio(n, len_per_block);
    max_num_blocks = std::max<int64_t>(max_num_blocks, num_blocks);
  }

  const int64_t total_elems = max_num_blocks * batch * k;
  const int64_t tmp_val_bytes =
      RoundUpTo(total_elems * kFloatBytes, kScratchAlignment);
  const int64_t tmp_idx_bytes =
      RoundUpTo(total_elems * kIndexBytes, kScratchAlignment);
  return tmp_val_bytes + tmp_idx_bytes;
}

// Bounds temporary workspace for RAFT select_radix (`sorted = true`):
//   Phase 1 (`raft::matrix::detail::select::radix::select_k` with BlockSize=512
//   and ItemsPerThread=32):
//     - When k == n: 0 bytes (direct copy + iota).
//     - One-block path (`n <= 16384` or `grid_dim == 1`):
//         `bufs` = chunk_one * buf_len * 2 * (sizeof(float) + sizeof(uint32_t))
//     - Multi-block path (`n > 16384` and `grid_dim > 1`):
//         `counters`   = Align256(chunk_multi * 640)
//         `histograms` = Align256(chunk_multi * num_buckets * sizeof(uint32_t))
//         `bufs`       = Align256(chunk_multi * buf_len * 2 * (4 + 4))
//       where `active_blocks <= 2048 / 512 = 4` and `num_waves = 10`.
//   Phase 2 (`offsets` + `segmented_sort_by_key` via CUB
//   `DeviceSegmentedRadixSort::SortPairs`):
//     - `offsets`        = Align256((batch + 1) * sizeof(uint32_t))
//     - `out_inds`       = Align256(batch * k * sizeof(uint32_t))
//     - `out_dists`      = Align256(batch * k * sizeof(float))
//     - `d_temp_storage` = Align256(batch * k * sizeof(float)) +
//                          Align256(batch * k * sizeof(uint32_t)) + 256
int64_t EstimateRadixScratchBytesFloat(RaftSelectKAlgo algo, int64_t batch,
                                       int64_t n, int64_t k, int64_t sm_count,
                                       RaftScratchPhaseMode phase_mode) {
  constexpr int64_t kRadixBlockSize = 512;
  constexpr int64_t kItemsPerThread = 32;
  constexpr int64_t kOneBlockMaxLen = kRadixBlockSize * kItemsPerThread;
  constexpr int64_t kMaxActiveBlocksPerSm = kMaxThreadsPerSm / kRadixBlockSize;
  constexpr int64_t kNumWaves = 10;
  constexpr int64_t kCounterBytes = 640;

  int64_t phase1_bytes = 0;
  if (k < n) {
    const uint32_t len_u32 = static_cast<uint32_t>(n);
    const int64_t one_block_buf_len = CalcRadixBufLenFloat<unsigned>(len_u32);
    const int64_t one_block_row_buf_bytes =
        one_block_buf_len * 2 * (kFloatBytes + kIndexBytes);

    const int64_t max_one_block_chunk =
        std::min<int64_t>(batch, kNumWaves * sm_count * kMaxActiveBlocksPerSm);
    const int64_t phase1_one_block_bytes = RoundUpTo(
        max_one_block_chunk * one_block_row_buf_bytes, kScratchAlignment);

    if (n <= kOneBlockMaxLen) {
      phase1_bytes = phase1_one_block_bytes;
    } else {
      const int64_t multi_block_buf_len =
          std::max(one_block_buf_len, CalcRadixBufLenFloat<float>(len_u32));
      const int64_t multi_block_row_buf_bytes =
          multi_block_buf_len * 2 * (kFloatBytes + kIndexBytes);
      const int64_t num_buckets =
          (algo == RaftSelectKAlgo::kRadix8bits) ? 256 : 2048;
      const int64_t max_items_per_chunk = kNumWaves * sm_count *
                                          kMaxActiveBlocksPerSm *
                                          kRadixBlockSize * kItemsPerThread;
      const int64_t max_multi_block_chunk = std::min<int64_t>(
          batch, std::max<int64_t>(1, max_items_per_chunk / n));
      const int64_t counters_bytes =
          RoundUpTo(max_multi_block_chunk * kCounterBytes, kScratchAlignment);
      const int64_t histograms_bytes = RoundUpTo(
          max_multi_block_chunk * num_buckets * kIndexBytes, kScratchAlignment);
      const int64_t bufs_bytes = RoundUpTo(
          max_multi_block_chunk * multi_block_row_buf_bytes, kScratchAlignment);
      const int64_t phase1_multi_block_bytes =
          counters_bytes + histograms_bytes + bufs_bytes;
      phase1_bytes =
          std::max<int64_t>(phase1_one_block_bytes, phase1_multi_block_bytes);
    }
  }

  const int64_t offsets_bytes =
      RoundUpTo((batch + 1) * kIndexBytes, kScratchAlignment);
  const int64_t values_bytes =
      RoundUpTo(batch * k * kFloatBytes, kScratchAlignment);
  const int64_t indices_bytes =
      RoundUpTo(batch * k * kIndexBytes, kScratchAlignment);
  const int64_t cub_temp_bytes =
      values_bytes + indices_bytes + kScratchAlignment;
  const int64_t phase2_bytes =
      offsets_bytes + indices_bytes + values_bytes + cub_temp_bytes;

  if (phase_mode == RaftScratchPhaseMode::kSumAcrossPhases) {
    return phase1_bytes + phase2_bytes;
  }
  return std::max<int64_t>(phase1_bytes, phase2_bytes);
}

}  // namespace

int GetRaftSelectKMaxSmCount(
    stream_executor::CudaComputeCapability compute_capability,
    int reported_sm_count) {
  using CudaCC = stream_executor::CudaComputeCapability;
  int arch_max_sm_count = kDefaultRaftSelectKMaxSmCount;
  switch (compute_capability.major) {
    case CudaCC::kPascal:
      arch_max_sm_count = 60;  // GP100 full die (P100 = 56).
      break;
    case CudaCC::kVolta:
      arch_max_sm_count = 84;  // GV100 full die (V100 = 80, TU102 = 72).
      break;
    case CudaCC::kAmpere:
      arch_max_sm_count = 144;  // GA100 = 128, AD102 (8.9) = 144.
      break;
    case CudaCC::kHopper:
      arch_max_sm_count = 144;  // GH100 full die (H100/H200 = 132, MIG = 16).
      break;
    case CudaCC::kBlackwell:
      if (compute_capability.minor < 7) {
        // Full dual-die Blackwell (B200 MIG = 18, B200 = 148, GB200 = 152,
        // GB300 = 152, B300 = 158).
        arch_max_sm_count = 160;
      } else {
        // Rubin family (10.7+, e.g. Vera Rubin NVL72 = 212 SMs).
        arch_max_sm_count = kDefaultRaftSelectKMaxSmCount;
      }
      break;
    case CudaCC::kBlackwell_11:
      arch_max_sm_count = 160;
      break;
    case CudaCC::kBlackwell_12:
      arch_max_sm_count = 192;  // GB202 full die (RTX PRO 6000 = 188).
      break;
    default:
      arch_max_sm_count = kDefaultRaftSelectKMaxSmCount;
      break;
  }
  return std::max(reported_sm_count, arch_max_sm_count);
}

absl::string_view RaftSelectKAlgoName(RaftSelectKAlgo algo) {
  switch (algo) {
    case RaftSelectKAlgo::kAuto:
      return "kAuto";
    case RaftSelectKAlgo::kRadix8bits:
      return "kRadix8bits";
    case RaftSelectKAlgo::kRadix11bits:
      return "kRadix11bits";
    case RaftSelectKAlgo::kRadix11bitsExtraPass:
      return "kRadix11bitsExtraPass";
    case RaftSelectKAlgo::kWarpAuto:
      return "kWarpAuto";
    case RaftSelectKAlgo::kWarpImmediate:
      return "kWarpImmediate";
    case RaftSelectKAlgo::kWarpFiltered:
      return "kWarpFiltered";
    case RaftSelectKAlgo::kWarpDistributed:
      return "kWarpDistributed";
    case RaftSelectKAlgo::kWarpDistributedShm:
      return "kWarpDistributedShm";
  }
  return "kUnknown";
}

// ============================================================================
// choose_select_k_algorithm_float
//
// Purpose:
//   Heuristic-based selection of the optimal "select k" algorithm depending on
//   problem shape (rows, cols, k). The decision is based on benchmark data.
//
// How the heuristic is generated:
//
//   1. Build the benchmark module:
//        cd raft
//        ./build.sh libraft bench-prims
//
//   2. Collect performance data by running microbenchmarks:
//
//        From the RAFT project root:
//          ./cpp/build/bench/prims/MATRIX_BENCH \
//            --benchmark_filter=Select \
//            --benchmark_out_format=json \
//            --benchmark_out=select_k_times.json
//
//        Output:
//          - Benchmark results are written to `select_k_times.json`
//
//   3. Generate the heuristic using the provided notebook:
//
//        ./cpp/scripts/heuristics/select_k/generate_heuristic.ipynb
//
//        The notebook consumes `select_k_times.json`, analyzes performance
//        trade-offs, and produces the decision tree implemented here.
// ============================================================================

RaftSelectKAlgo choose_select_k_algorithm_float(uint32_t rows, uint32_t cols,
                                                uint32_t k) {
  if (k > 163) {
    if (rows > 803) {
      if (k > 256) {
        return RaftSelectKAlgo::kRadix8bits;
      } else {
        if (cols > 3120) {
          return RaftSelectKAlgo::kWarpDistributedShm;
        } else {
          return RaftSelectKAlgo::kWarpFiltered;
        }
      }
    } else {
      if (cols > 19332) {
        if (cols > 106382976) {
          return RaftSelectKAlgo::kRadix11bitsExtraPass;
        } else {
          if (k > 256) {
            return RaftSelectKAlgo::kRadix11bits;
          } else {
            if (rows > 384) {
              return RaftSelectKAlgo::kWarpDistributedShm;
            } else {
              if (cols > 110946) {
                return RaftSelectKAlgo::kRadix11bits;
              } else {
                return RaftSelectKAlgo::kWarpFiltered;
              }
            }
          }
        }
      } else {
        if (k > 256) {
          return RaftSelectKAlgo::kRadix8bits;
        } else {
          return RaftSelectKAlgo::kWarpFiltered;
        }
      }
    }
  } else {
    if (k > 1) {
      if (cols > 34520) {
        if (k > 73) {
          if (rows > 132) {
            return RaftSelectKAlgo::kWarpDistributedShm;
          } else {
            if (cols > 259191) {
              if (cols > 12021888) {
                if (rows > 2) {
                  return RaftSelectKAlgo::kWarpDistributedShm;
                } else {
                  return RaftSelectKAlgo::kRadix11bits;
                }
              } else {
                if (rows > 36) {
                  return RaftSelectKAlgo::kWarpDistributedShm;
                } else {
                  return RaftSelectKAlgo::kRadix11bits;
                }
              }
            } else {
              if (rows > 48) {
                return RaftSelectKAlgo::kRadix11bits;
              } else {
                return RaftSelectKAlgo::kWarpFiltered;
              }
            }
          }
        } else {
          if (k > 4) {
            return RaftSelectKAlgo::kWarpDistributedShm;
          } else {
            if (rows > 18) {
              return RaftSelectKAlgo::kWarpDistributedShm;
            } else {
              if (cols > 1652995) {
                return RaftSelectKAlgo::kWarpDistributedShm;
              } else {
                return RaftSelectKAlgo::kWarpImmediate;
              }
            }
          }
        }
      } else {
        if (rows > 363) {
          if (cols > 1115) {
            return RaftSelectKAlgo::kWarpDistributedShm;
          } else {
            if (rows > 1536) {
              return RaftSelectKAlgo::kWarpDistributedShm;
            } else {
              return RaftSelectKAlgo::kWarpImmediate;
            }
          }
        } else {
          if (k > 85) {
            if (cols > 1034) {
              return RaftSelectKAlgo::kWarpFiltered;
            } else {
              return RaftSelectKAlgo::kWarpImmediate;
            }
          } else {
            if (rows > 81) {
              if (k > 4) {
                if (cols > 12288) {
                  return RaftSelectKAlgo::kWarpDistributedShm;
                } else {
                  return RaftSelectKAlgo::kWarpImmediate;
                }
              } else {
                return RaftSelectKAlgo::kWarpImmediate;
              }
            } else {
              return RaftSelectKAlgo::kWarpImmediate;
            }
          }
        }
      }
    } else {
      return RaftSelectKAlgo::kWarpImmediate;
    }
  }
}

absl::StatusOr<int64_t> EstimateRaftSelectKScratchBytesFloat(
    uint32_t batch, uint32_t n, uint32_t k, int sm_count,
    RaftScratchPhaseMode phase_mode) {
  if (batch == 0 || n == 0 || k == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RAFT select_k dimensions must be positive, got batch=", batch,
        ", n=", n, ", k=", k));
  }
  if (k > n) {
    return absl::InvalidArgumentError(
        absl::StrCat("RAFT select_k requires k <= n, got k=", k, ", n=", n));
  }
  constexpr uint32_t kMaxInt32 =
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  if (batch > kMaxInt32) {
    return absl::InvalidArgumentError(
        absl::StrCat("RAFT select_k batch exceeds int32_t max: batch=", batch));
  }
  if (n > kMaxInt32) {
    return absl::InvalidArgumentError(
        absl::StrCat("RAFT select_k n exceeds int32_t max: n=", n));
  }
  if (uint64_t{batch} * uint64_t{k} >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RAFT select_k batch * k exceeds uint32_t max: batch=", batch,
        ", k=", k));
  }

  const int64_t effective_sm_count =
      sm_count > 0 ? static_cast<int64_t>(sm_count)
                   : static_cast<int64_t>(kDefaultRaftSelectKMaxSmCount);
  const RaftSelectKAlgo algo = choose_select_k_algorithm_float(batch, n, k);
  switch (algo) {
    case RaftSelectKAlgo::kWarpImmediate:
    case RaftSelectKAlgo::kWarpFiltered:
    case RaftSelectKAlgo::kWarpDistributed:
    case RaftSelectKAlgo::kWarpDistributedShm:
      return AddSafetyMargin(
          EstimateWarpSortScratchBytesFloat(batch, n, k, effective_sm_count));
    case RaftSelectKAlgo::kRadix8bits:
    case RaftSelectKAlgo::kRadix11bits:
    case RaftSelectKAlgo::kRadix11bitsExtraPass:
      return AddSafetyMargin(EstimateRadixScratchBytesFloat(
          algo, batch, n, k, effective_sm_count, phase_mode));
    case RaftSelectKAlgo::kAuto:
    case RaftSelectKAlgo::kWarpAuto:
      break;
  }
  return absl::InternalError(
      absl::StrCat("Unexpected RaftSelectKAlgo: ", algo));
}

}  // namespace xla::gpu

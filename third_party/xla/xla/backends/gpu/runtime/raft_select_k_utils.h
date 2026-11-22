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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RAFT_SELECT_K_UTILS_H_
#define XLA_BACKENDS_GPU_RUNTIME_RAFT_SELECT_K_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace xla::gpu {

// Conservative default upper bound on GPU streaming multiprocessor (SM) count
// when a target DeviceDescription's core_count is unavailable or uninitialized.
inline constexpr int kDefaultRaftSelectKMaxSmCount = 256;

// Returns a conservative full-die streaming multiprocessor (SM) count upper
// bound for `compute_capability` (at least `reported_sm_count`).
//
// Sizing RAFT workspace to the architecture family's full-die SM ceiling keeps
// AOT and cached executables safe when compiled on a MIG slice or lower-SM SKU
// (e.g. B200 MIG = 18 SMs, B200 = 148 SMs, GB300 = 152 SMs) and executed on a
// compatible higher-SM SKU in the same family (e.g. GB200 = 152 SMs,
// B300 = 158 SMs, up to 160 physical SMs on full Blackwell dies).
int GetRaftSelectKMaxSmCount(
    stream_executor::CudaComputeCapability compute_capability,
    int reported_sm_count = 0);

// XLA-internal mirror of raft::matrix::SelectAlgo so host-only compiler passes
// and utilities can select RAFT Top-K algorithms without depending on CUDA-only
// RAFT headers.
enum class RaftSelectKAlgo : uint8_t {
  kAuto = 0,
  kRadix8bits = 1,
  kRadix11bits = 2,
  kRadix11bitsExtraPass = 3,
  kWarpAuto = 4,
  kWarpImmediate = 5,
  kWarpFiltered = 6,
  kWarpDistributed = 7,
  kWarpDistributedShm = 8,
};

// Controls how two-phase Radix select_k scratch sizes are combined.
enum class RaftScratchPhaseMode : uint8_t {
  // Sizes Radix scratch to max(Phase1, Phase2), matching runtimes where
  // FixedBufferDeviceMemoryResource reclaims Phase 1 buffers before Phase 2.
  kMaxAcrossPhases = 0,
  // Sizes Radix scratch to Phase1 + Phase2 for forward compatibility with
  // older runtimes whose bump allocator did not reset between phases.
  kSumAcrossPhases = 1,
};

absl::string_view RaftSelectKAlgoName(RaftSelectKAlgo algo);

template <typename Sink>
void AbslStringify(Sink& sink, RaftSelectKAlgo algo) {
  sink.Append(RaftSelectKAlgoName(algo));
}

// Heuristic-based selection of the RAFT select_k algorithm for float32 inputs
// of shape [rows, cols] selecting top-k elements per row.
RaftSelectKAlgo choose_select_k_algorithm_float(uint32_t rows, uint32_t cols,
                                                uint32_t k);

// Heuristic-based selection of the RAFT select_k algorithm for bfloat16 inputs
// of shape [rows, cols] selecting top-k elements per row.
RaftSelectKAlgo choose_select_k_algorithm_bfloat16(uint32_t rows, uint32_t cols,
                                                   uint32_t k);

// Heuristic-based selection of the RAFT select_k algorithm for uint64_t inputs
// of shape [rows, cols] selecting top-k elements per row.
RaftSelectKAlgo choose_select_k_algorithm_uint64(uint32_t rows, uint32_t cols,
                                                 uint32_t k);

// Calculates a 256-byte-aligned upper bound (in bytes) on the temporary
// workspace required by RAFT select_k (`__gpu$TopK`) for float32 inputs of
// shape [batch, n] selecting top-k sorted elements per row.
//
// Requires `batch > 0`, `n > 0`, `k <= n`, `batch <= INT32_MAX`,
// `n <= INT32_MAX` (matching XLA's signed S32 TopK output indices and RAFT's
// 32-bit signed launch arithmetic), and `batch * k <= UINT32_MAX`.
//
// Uses `choose_select_k_algorithm_float(batch, n, k)` to determine the planned
// RAFT algorithm, bounds occupancy-dependent temporary buffers across all
// valid CUDA thread-block configurations for `sm_count` streaming
// multiprocessors (falling back to `kDefaultRaftSelectKMaxSmCount` when
// `sm_count <= 0`), and adds a `max(4 KB, ~6.25%)` safety margin aligned to
// 256 bytes to absorb base-pointer alignment padding and minor library
// metadata changes across GPU AOT runtime versions.
absl::StatusOr<int64_t> EstimateRaftSelectKScratchBytesFloat(
    uint32_t batch, uint32_t n, uint32_t k,
    int sm_count = kDefaultRaftSelectKMaxSmCount,
    RaftScratchPhaseMode phase_mode = RaftScratchPhaseMode::kMaxAcrossPhases);

// Calculates a 256-byte-aligned upper bound (in bytes) on the temporary
// workspace required by RAFT select_k (`__gpu$TopK`) for bfloat16 inputs of
// shape [batch, n] selecting top-k sorted elements per row.
//
// Uses `choose_select_k_algorithm_bfloat16(batch, n, k)` with the same
// dimension preconditions, SM occupancy upper bounds, and safety margin as
// `EstimateRaftSelectKScratchBytesFloat`.
absl::StatusOr<int64_t> EstimateRaftSelectKScratchBytesBFloat16(
    uint32_t batch, uint32_t n, uint32_t k,
    int sm_count = kDefaultRaftSelectKMaxSmCount,
    RaftScratchPhaseMode phase_mode = RaftScratchPhaseMode::kMaxAcrossPhases);

// Calculates a 256-byte-aligned upper bound (in bytes) on the temporary
// workspace required by RAFT select_k (`__gpu$TopK`) for uint64_t inputs of
// shape [batch, n] selecting top-k sorted elements per row.
//
// Uses `choose_select_k_algorithm_uint64(batch, n, k)` with the same dimension
// preconditions, SM occupancy upper bounds, and safety margin as
// `EstimateRaftSelectKScratchBytesFloat`.
absl::StatusOr<int64_t> EstimateRaftSelectKScratchBytesUInt64(
    uint32_t batch, uint32_t n, uint32_t k,
    int sm_count = kDefaultRaftSelectKMaxSmCount,
    RaftScratchPhaseMode phase_mode = RaftScratchPhaseMode::kMaxAcrossPhases);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAFT_SELECT_K_UTILS_H_

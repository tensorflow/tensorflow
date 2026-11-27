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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns the embedded KernelLoaderSpec for the unified SingleTile radix sort
// kernel (handling both unbatched and batched segments whose segment size fits
// within `CubRadixSortPolicySpec::single_tile.tile_items()`).
//
// Prefers the single-architecture CUBIN ELF matching `compute_capability`.
// When no matching CUBIN is embedded (for example, PTX-only builds), falls back
// to the fatbinary only if the highest JIT-eligible PTX entry maps to the same
// canonical policy tier as `CanonicalCubSortPolicySmVersion(cc.major)`.
//
// Kernel ABI:
// - Keys-only (arity 3): `(const Key* __restrict__ keys_in,
//                          Key* __restrict__ keys_out,
//                          int32_t segment_size)`
// - Pairs (arity 5):     `(const Key* __restrict__ keys_in,
//                          Key* __restrict__ keys_out,
//                          const Value* __restrict__ values_in,
//                          Value* __restrict__ values_out,
//                          int32_t segment_size)`
// Launch geometry: `BlockDim(num_segments)`,
// `ThreadDim(policy.single_tile.block_threads)`, with
// `0 <= segment_size <= policy.single_tile.tile_items()`. Input and output
// buffers must be non-aliasing (`__restrict__`).
absl::StatusOr<stream_executor::KernelLoaderSpec>
GetCubSortSingleTileKernelSpec(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability);

namespace internal {

// Verifies that `fatbin` contains a JIT-eligible PTX entry for
// `compute_capability` whose highest matching PTX arch maps to the same
// canonical SM policy tier as `CanonicalCubSortPolicySmVersion(cc.major)`.
// Exposed for unit tests.
absl::Status ValidateFatbinPtxPolicyTier(
    absl::Span<const uint8_t> fatbin,
    const stream_executor::CudaComputeCapability& compute_capability);

}  // namespace internal

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_H_

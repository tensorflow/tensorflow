/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
#define XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace experimental {

// Evaluates a subset of Triton hardware and emitter constraints on candidate
// tile sizes without requiring construction of concrete tiled HLO instructions.
// This allows us to prune invalid tiling candidates early, but doesn't replace
// the full verification in `VerifyTritonConstraints`.
Decision VerifySubsetOfTritonConstraints(
    absl::Span<const int64_t> padded_tile_sizes,
    const TilingSpace& tiling_space, const se::DeviceDescription& device_info);

Decision VerifyTritonConstraints(const TiledHloComputation& tiled_computation,
                                 const se::DeviceDescription& device_info);

}  // namespace experimental

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_

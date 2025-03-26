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

#ifndef XLA_SERVICE_COLLECTIVE_UTILS_H_
#define XLA_SERVICE_COLLECTIVE_UTILS_H_

#include <cstdint>

namespace xla {

// Defines the default threshold for `AllReduceCombiner` up to which the pass
// will combine collectives.
constexpr int64_t kDefaultAllReduceCombineThreshold = 30 * 1024 * 1024 + 7;

// Defines the default threshold for `AllGatherCombiner` up to which the pass
// will combine collectives.
constexpr int64_t kDefaultAllGatherCombineThreshold = 30 * 1024 * 1024 + 7;

// Defines the default threshold for `CollectivePermuteCombiner` up to which the
// pass will combine collectives.
constexpr int64_t kDefaultCollectivePermuteCombineThreshold =
    30 * 1024 * 1024 + 7;

// Defines the default threshold for `ReduceScatterCombiner` up to which the
// pass will combine collectives.
constexpr int64_t kDefaultReduceScatterCombineThreshold = 30 * 1024 * 1024 + 7;

// Defines the default coefficient for the SoL NCCL collective cost model.
// Note: XLA flags allow a user to override the default values of the model.
constexpr float kDefaultNcclCostModelCoeff = 0.45f;
// Chunk size is 4MiBytes (4*1024*1024 bytes)
constexpr int64_t kDefaultNcclCostModelChunkSizeBytes = 4194304;
constexpr int64_t kDefaultNcclCostModelGPUsPerNode = 8;
}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_UTILS_H_

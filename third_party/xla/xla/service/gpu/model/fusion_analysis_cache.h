/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_FUSION_ANALYSIS_CACHE_H_
#define XLA_SERVICE_GPU_MODEL_FUSION_ANALYSIS_CACHE_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Caches HloFusionAnalyses. Thread-compatible, if no threads concurrently `Get`
// and `Invalidate` the same key. Analyses are cached based on unique_ids, no
// checking or tracking of changes is done.
class HloFusionAnalysisCache {
 public:
  explicit HloFusionAnalysisCache(
      const stream_executor::DeviceDescription& device_info)
      : device_info_(device_info) {}

  // Returns the analysis for the given instruction, creating it if it doesn't
  // exist yet. Do not call concurrently with `Invalidate` for the same key.
  const HloFusionAnalysis& Get(const HloInstruction& instruction);

  // Returns the analysis for the given producer/consumer pair.
  const HloFusionAnalysis& Get(const HloInstruction& producer,
                               const HloInstruction& consumer);

  // Removes the cache entry for the given instruction, if it exists. Also
  // removes all producer-consumer fusions that involve this instruction.
  void Invalidate(const HloInstruction& instruction);

  // Delete all cache entries.
  void Clear();

 private:
  const stream_executor::DeviceDescription& device_info_;

  absl::Mutex mutex_;

  // All `int` keys and values here are unique instruction IDs.
  absl::node_hash_map<int, HloFusionAnalysis> analyses_;
  absl::node_hash_map<std::pair<int, int>, HloFusionAnalysis>
      producer_consumer_analyses_;

  // For each instruction `producer`, contains the `consumer`s for which we have
  // entries {`producer`, `consumer`} in `producer_consumer_analyses_`.
  absl::flat_hash_map<int, std::vector<int>> consumers_for_producers_;
  // For each instruction `consumer`, contains the `producer`s for which we have
  // entries {`producer`, `consumer`} in `producer_consumer_analyses_`.
  absl::flat_hash_map<int, std::vector<int>> producers_for_consumers_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_FUSION_ANALYSIS_CACHE_H_

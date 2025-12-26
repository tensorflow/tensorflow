/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_OPTIONS_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_OPTIONS_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"

namespace xla {

struct HloRematerializationOptions {
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;

  using CompactShapeFunction =
      std::function<absl::StatusOr<Shape>(const Shape&)>;

  // The high-level rematerialization algorithm to use. Will significantly
  // affect the runtime performance of the pass and the memory utilization of
  // the HLO module.
  enum class RematAlgorithm {
    kAlwaysRemat,   // Rematerializes anything it can at any point the HLO is
                    // above the memory limit. Default rematerialization
                    // algorithm.
    kPeakPriority,  // Prioritize rematerializing the highest peak in the module
                    // at any given step. Much slower than the default algorithm
                    // but can offer better memory utilization in certain cases.
  };

  // Mode in which the rematerialization algorithm should be run.
  struct RematerializationModeConfig {
    RematerializationModeConfig(bool recompute, bool compress,
                                bool host_offload)
        : recompute(recompute),
          compress(compress),
          host_offload(host_offload) {}
    bool recompute;     // Enables the kRecompute RematStrategy.
    bool compress;      // Enables the kCompress RematStrategy.
    bool host_offload;  // Enables the kHostOffload RematStrategy.
  };

  // This is a struct containing configuration options that are specific to the
  // Host Memory Offload strategy.
  struct HostMemoryOffloadConfig {
    explicit HostMemoryOffloadConfig(int64_t host_memory_space,
                                     float bandwidth_to_host_bytes_per_second,
                                     float bandwidth_from_host_bytes_per_second)
        : host_memory_space(host_memory_space),
          bandwidth_to_host_bytes_per_second(
              bandwidth_to_host_bytes_per_second),
          bandwidth_from_host_bytes_per_second(
              bandwidth_from_host_bytes_per_second) {}

    // The host memory space, which is used during the host offload strategy.
    int64_t host_memory_space;

    float bandwidth_to_host_bytes_per_second;

    float bandwidth_from_host_bytes_per_second;
  };

  static Shape DefaultCompactShapeFunction(const Shape& shape) { return shape; }

  explicit HloRematerializationOptions(
      HloCostAnalysis& hlo_cost_analysis,
      const RematerializationModeConfig& remat_mode_config,
      int64_t memory_limit_bytes, int block_size_limit,
      float block_rematerialization_factor, int64_t min_remat_size,
      CompactShapeFunction compact_shape_function,
      std::optional<HostMemoryOffloadConfig> host_memory_offload_config =
          std::nullopt,
      absl::flat_hash_map<HloComputation*, int64_t>
          async_computation_parallelism = {},
      RematAlgorithm remat_algorithm = RematAlgorithm::kAlwaysRemat)
      : hlo_cost_analysis(hlo_cost_analysis),
        remat_mode_config(remat_mode_config),
        memory_limit_bytes(memory_limit_bytes),
        block_size_limit(block_size_limit),
        block_rematerialization_factor(block_rematerialization_factor),
        min_remat_size(min_remat_size),
        compact_shape_function(compact_shape_function == nullptr
                                   ? DefaultCompactShapeFunction
                                   : std::move(compact_shape_function)),
        host_memory_offload_config(host_memory_offload_config),
        async_computation_parallelism(std::move(async_computation_parallelism)),
        remat_algorithm(remat_algorithm) {}

  // The cost model used for decisions during rematerialization for host
  // memory offload. It is also used for getting Shape size.
  HloCostAnalysis& hlo_cost_analysis;

  // Holds the rematerialization strategy configuration to be used by the
  // pass.
  RematerializationModeConfig remat_mode_config;

  // Function which computes the size of the top-level buffer of a shape.
  const ShapeSizeFunction size_function;

  // The threshold number of bytes to reduce memory use to via
  // rematerialization. Size of aliased outputs should be subtracted
  // from this.
  int64_t memory_limit_bytes;

  // Maximum number of consecutive instructions to consider for
  // rematerialization.
  int block_size_limit;

  // Controls the amount of effort spent trying to find large blocks for
  // rematerialization. Larger values leads to longer compilation times in
  // return for potentially reduced memory consumption.
  float block_rematerialization_factor;

  // The minimum size, in bytes, of a tensor to be considered for
  // rematerialization. All tensors smaller than this size will be skipped
  // over.
  int64_t min_remat_size;

  // Converts a shape into compact form, returns the same shape if a shape is
  // already considered compact.
  CompactShapeFunction compact_shape_function;

  std::optional<HostMemoryOffloadConfig> host_memory_offload_config;

  // Collection of async entry computations and their number of parallel
  // invocations.
  absl::flat_hash_map<HloComputation*, int64_t> async_computation_parallelism;

  // The high-level rematerialization algorithm to be used.
  RematAlgorithm remat_algorithm;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_OPTIONS_H_

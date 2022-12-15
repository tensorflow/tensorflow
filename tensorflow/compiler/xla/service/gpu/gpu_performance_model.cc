/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_performance_model.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/time/time.h"

namespace xla {
namespace gpu {

/*static*/ struct GpuPerformanceModel::RunTimes
GpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    const GpuDeviceInfo& gpu_device_info,
    const std::vector<HloInstruction*> fused_users, bool multi_output) {
  VLOG(8) << "Producer: " << producer->name();
  VLOG(10) << producer->fused_instructions_computation()->ToString();

  float memory_bandwidth_bytes_per_second = gpu_device_info.memory_bandwidth;
  int64_t l2_cache_size = gpu_device_info.l2_cache_size;
  int64_t l1_cache_size = kL1CacheSizePerSM * gpu_device_info.core_count;

  // Account for L1 / L2 cache speedup if input is small and read multiple
  // times.
  auto read_time = [&](int64_t n_bytes_net, int64_t n_bytes_with_repeats) {
    float bw = memory_bandwidth_bytes_per_second;
    if (n_bytes_net < l2_cache_size) {
      bw *= kL2CacheSpeedup;
      if (n_bytes_net < l1_cache_size) {
        bw *= kL1CacheSpeedup;
      }
    }
    return absl::Seconds(n_bytes_with_repeats / bw);
  };

  auto producer_input_access_time = [&](float output_utilization) {
    // Assume that accessed input sizes scale linearly with the utilization
    // of the output. TODO(sergachev): Run this through the HLO cost
    // analysis for a more accurate estimate.
    absl::Duration ret = absl::ZeroDuration();
    for (int i = 0; i < producer->operand_count(); ++i) {
      int64_t p_size_accessed =
          cost_analysis->operand_bytes_accessed(*producer, i);
      float operand_utilization =
          cost_analysis->operand_utilization(*producer, i);
      int64_t p_size_net = 0;
      if (operand_utilization != 0) {
        p_size_net = static_cast<float>(p_size_accessed) / operand_utilization;
      }
      ret += read_time(std::min(p_size_net, p_size_accessed),
                       p_size_accessed * output_utilization);
    }
    return ret;
  };

  float producer_bytes_out = cost_analysis->output_bytes_accessed(*producer);
  float producer_bytes_in =
      cost_analysis->bytes_accessed(*producer) - producer_bytes_out;
  VLOG(8) << "Producer FLOPs: " << cost_analysis->flop_count(*producer);
  VLOG(8) << "Producer bytes in: " << producer_bytes_in;
  VLOG(8) << "Producer bytes out: " << producer_bytes_out;
  float producer_elements_out =
      ShapeUtil::ElementsInRecursive(producer->shape());
  VLOG(8) << "Producer elements out: " << producer_elements_out;

  auto compute_time = [&](int64_t n_flops, int64_t n_threads) {
    int fpu_count = gpu_device_info.core_count * gpu_device_info.fpus_per_core;
    float n_threads_active = fmin(n_threads, fpu_count);
    float flop_per_second_per_fpu = 2 * 1e9 * gpu_device_info.clock_rate_ghz;
    float flop_per_second_effective =
        flop_per_second_per_fpu * n_threads_active;
    return absl::Seconds(n_flops / flop_per_second_effective);
  };

  absl::Duration compute_time_unfused =
      compute_time(cost_analysis->flop_count(*producer), producer_elements_out);
  VLOG(8) << "Compute time unfused: " << compute_time_unfused;
  VLOG(8) << "Input access time unfused: " << producer_input_access_time(1.0);
  absl::Duration output_write_time_unfused =
      absl::Seconds(producer_bytes_out / memory_bandwidth_bytes_per_second);
  VLOG(8) << "Output write time unfused: " << output_write_time_unfused;
  absl::Duration exec_time_unfused =
      std::max(compute_time_unfused,
               producer_input_access_time(1.0) + output_write_time_unfused);

  int64_t fused_consumer_count = fused_users.size();
  VLOG(8) << "Consumer count: " << fused_consumer_count;
  float total_producer_utilization = 0;

  absl::Duration exec_time_fused = absl::ZeroDuration();
  absl::Duration producer_output_read_time_unfused = absl::ZeroDuration();
  for (const HloInstruction* u : fused_users) {
    float utilization_by_this_consumer =
        cost_analysis->operand_utilization(*u, u->operand_index(producer));
    total_producer_utilization += utilization_by_this_consumer;
    absl::Duration compute_time_by_this_consumer = compute_time(
        cost_analysis->flop_count(*producer) * utilization_by_this_consumer,
        producer_elements_out * utilization_by_this_consumer);
    exec_time_fused +=
        std::max(compute_time_by_this_consumer,
                 producer_input_access_time(utilization_by_this_consumer));
    producer_output_read_time_unfused +=
        read_time(std::min(producer_bytes_out,
                           producer_bytes_out * utilization_by_this_consumer),
                  producer_bytes_out * utilization_by_this_consumer);
  }
  VLOG(8) << "Utilization of producer output: " << total_producer_utilization;

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumer_count + 1) + exec_time_unfused +
      producer_output_read_time_unfused;
  VLOG(8) << "Unfused time: " << time_unfused;

  absl::Duration time_fused =
      kKernelLaunchOverhead * fused_consumer_count + exec_time_fused;
  // Multi-output fusion still writes the initial output of the producer.
  // For now assume that the producer's output does not need to be recomputed.
  if (multi_output) {
    time_fused += output_write_time_unfused;
  }
  VLOG(8) << "Fused time: " << time_fused;

  return {time_unfused, time_fused};
}

}  // namespace gpu
}  // namespace xla

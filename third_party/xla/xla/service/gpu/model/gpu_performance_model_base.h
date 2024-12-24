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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_BASE_H_
#define XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_BASE_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct EstimateRunTimeData {
  int64_t flops;
  int64_t bytes_read;
  int64_t bytes_written;
  absl::Duration read_time;
  absl::Duration write_time;
  absl::Duration compute_time;
  absl::Duration exec_time;

  // Returns an estimate that is guaranteed to be zero.
  static EstimateRunTimeData Zero() {
    return EstimateRunTimeData{/*flops=*/0,
                               /*bytes_read=*/0,
                               /*bytes_written=*/0,
                               /*read_time=*/absl::ZeroDuration(),
                               /*write_time=*/absl::ZeroDuration(),
                               /*compute_time=*/absl::ZeroDuration(),
                               /*exec_time=*/absl::ZeroDuration()};
  }

  // Returns an estimate that is guaranteed to be larger than any real runtime.
  static EstimateRunTimeData Infinite() {
    return EstimateRunTimeData{
        /*flops=*/std::numeric_limits<int64_t>::max(),
        /*bytes_read=*/std::numeric_limits<int64_t>::max(),
        /*bytes_written=*/std::numeric_limits<int64_t>::max(),
        /*read_time=*/absl::InfiniteDuration(),
        /*write_time=*/absl::InfiniteDuration(),
        /*compute_time=*/absl::InfiniteDuration(),
        /*exec_time=*/absl::InfiniteDuration()};
  }

  // Returns true if the estimate is guaranteed to be larger than any real
  // runtime.
  bool IsInfinite() const { return exec_time == absl::InfiniteDuration(); }

  std::string ToString() const {
    return absl::StrFormat(
        "EstimateRunTimeData{\n"
        " flops: %d\n"
        " bytes_read: %d\n"
        " bytes_written: %d\n"
        " read_time: %s\n"
        " write_time: %s\n"
        " compute_time: %s\n"
        " exec_time: %s\n"
        "}",
        flops, bytes_read, bytes_written, absl::FormatDuration(read_time),
        absl::FormatDuration(write_time), absl::FormatDuration(compute_time),
        absl::FormatDuration(exec_time));
  }
};

class GpuPerformanceModelCache {
 public:
  // Returns cached runtime data for the instruction or producer-consumer pair.
  // Returns nullopt if there is no data in cache.
  std::optional<EstimateRunTimeData> Get(const HloInstruction& instruction);
  std::optional<absl::Duration> Get(const HloInstruction& producer,
                                    const HloInstruction& consumer);
  const absl::flat_hash_map<const HloInstruction*, absl::Duration>&
  // Returns cache entries for all consumers of this producer.
  GetAllConsumers(const HloInstruction& producer);
  // Checks if producer-consumer pair cache entries exist for this producer.
  bool ContainsConsumers(const HloInstruction& producer);
  // Sets cache value for the instruction or producer-consumer pair.
  void Set(const HloInstruction& instruction,
           const EstimateRunTimeData& runtime_data);
  void Set(const HloInstruction& producer, const HloInstruction& consumer,
           absl::Duration runtime);

  // Removes all cache entries for this instruction. The cache contains entries
  // for individual instructions in instruction_runtime_data_ and for
  // producer-consumer pairs in fusion_runtime_data_.
  void Invalidate(const HloInstruction& instruction);

 private:
  absl::Mutex mutex_;

  // Stores unfused runtime data for individual instructions.
  absl::flat_hash_map<const HloInstruction*, EstimateRunTimeData>
      instruction_runtime_data_;

  // Stores fused runtime data for producer-consumer pairs.
  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<const HloInstruction*, absl::Duration>>
      fusion_runtime_data_;
};

struct GpuPerformanceModelOptions {
  // Factor for how much parallelism between compute and memory accesses should
  // be assumed. If 1.0, assume perfect parallelism (the run time is the maximum
  // of both times). If 0.0, assume no parallelism (the run time is the sum of
  // both times).
  //
  // This constant was chosen empirically in early 2024, based on runtime
  // performance on a set of benchmarks internal to Google. Intuitively, we
  // expect it to be close to 1, but not quite 1 (i.e., sometimes, compute
  // or memory accesses will be stalled waiting for the other, but usually
  // they won't).
  double memory_compute_parallelism = 0.95;

  // If present, use this to retrieve fusion analyses.
  HloFusionAnalysisCache* fusion_analysis_cache = nullptr;

  GpuPerformanceModelCache* gpu_performance_model_cache = nullptr;

  static GpuPerformanceModelOptions Default(
      HloFusionAnalysisCache* fusion_analysis_cache = nullptr,
      GpuPerformanceModelCache* gpu_performance_model_cache = nullptr) {
    GpuPerformanceModelOptions config;
    config.fusion_analysis_cache = fusion_analysis_cache;
    config.gpu_performance_model_cache = gpu_performance_model_cache;
    return config;
  }
};

class GpuPerformanceModelBase {
 public:
  struct RunTimes {
    absl::Duration time_unfused;
    absl::Duration time_fused;
  };

  // Estimated values in the absence of easy ways to query them.
  static constexpr absl::Duration kKernelLaunchOverhead = absl::Microseconds(1);
  static constexpr absl::Duration kNcclKernelLaunchOverhead =
      absl::Microseconds(5);
  static constexpr float kL2CacheSpeedup = 2.5;
  static constexpr float kL1CacheSpeedup = 8;

  // Uses HloFusionAnalysis for computing the actual number of threads and
  // blocks that the IR emitter will use.
  static LaunchDimensions EstimateFusionLaunchDimensions(
      const HloFusionAnalysis& fusion_analysis);

  // Returns bytes accessed of operand output by instruction. Returns 0, if the
  // operand is not used by the instruction.
  static int64_t GetOperandBytesAccessed(
      const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
      const HloInstruction* operand);

  // Returns utilization of operand by instruction. Returns 0, if the operand is
  // not used by the instruction.
  static float GetOperandUtilization(const GpuHloCostAnalysis* cost_analysis,
                                     const HloInstruction* instr,
                                     const HloInstruction* operand);

  // Returns utilization `overlap` between a common operand of producer and
  // consumer on merge. `utilization > 0` means that the operand will be
  // accessed more efficiently after fusion.
  //
  // Currently covers two cases:
  // 1) Producer has to use the common operand elementwise from its root if it
  //    is a fusion or just be an elementwise instruction.
  // 2) Consumer has to have common elementwise roots for the producer and the
  //    common operand if it is a fusion or just be an elementwise instruction.
  static float GetCommonUtilization(const GpuHloCostAnalysis* cost_analysis,
                                    const HloInstruction* producer,
                                    int64_t producer_idx_of_operand,
                                    const HloInstruction* consumer);

  // Returns bytes accessed of operand after producer and consumer are fused
  // together. `GetCommonUtilization` works only for a limited set of
  // elementwise cases.
  static int64_t GetSharedOperandBytesAccessed(
      const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
      const HloInstruction* consumer, const HloInstruction* operand);

  // Estimate read time of n_bytes_total bytes from global memory on a
  // given GPU.
  //
  // Assumes that the first n_bytes_net are always read from DRAM, but next
  // reads can be cached. Restricts the effective HBM bandwidth using the
  // utilization rate passed as a parameter to model not-fully-coalesced reads.
  static absl::Duration ReadTimeWithDRAMHeuristic(
      const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
      int64_t n_bytes_net, int64_t n_bytes_total, PrimitiveType element_type,
      double hbm_bandwidth_utilization_rate);

  static absl::Duration WriteTime(const se::DeviceDescription& gpu_device_info,
                                  int64_t bytes_written);

  static absl::Duration ComputeTime(
      const se::DeviceDescription& gpu_device_info, int64_t flops,
      int64_t num_blocks, int64_t num_threads_per_block);

  static absl::Duration CombineComputeAndMemoryAccessTime(
      absl::Duration compute_time, absl::Duration memory_access_time,
      const GpuPerformanceModelOptions& config);

  // Logs estimates for the operand read if VLOG is enabled.
  static void VLogOperandRead(const HloInstruction* operand,
                              int64_t n_bytes_total, int64_t n_bytes_net,
                              bool coalesced);
};

// Given an element type and whether the read is coalesced, returns the
// utilization rate of the HBM bandwidth.
//
// TODO(b/332714755): to avoid interfering with the cost model as it exists
// right now, this duplicates pre-existing logic and doesn't take into account
// how much of the memory access is actually useful and just assumes the worst
// possible utilization if the read is uncoalesced.
double GetCoalescingUtilizationRate(
    PrimitiveType element_type, const se::DeviceDescription& gpu_device_info,
    bool coalesced);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_BASE_H_

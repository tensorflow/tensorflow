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

#include "xla/service/gpu/model/combined_gpu_performance_model.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

CombinedGpuPerformanceModel::CombinedGpuPerformanceModel(
    const se::DeviceDescription& device_info,
    HloFusionAnalysisCache& fusion_analysis_cache,
    mlir::MLIRContext& mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size)
    : device_info_(device_info),
      fusion_analysis_cache_(fusion_analysis_cache),
      mlir_context_(mlir_context),
      indexing_model_(&device_info, &fusion_analysis_cache, shape_size,
                      &mlir_context),
      model_(device_info, fusion_analysis_cache, cache_, &mlir_context) {}

absl::StatusOr<EstimateRunTimeData>
CombinedGpuPerformanceModel::EstimateRunTimeForInstruction(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  {
    {
      absl::MutexLock lock(cache_mutex_);
      if (auto cached = cache_.Get(*instr)) {
        return *cached;
      }
    }

    absl::StatusOr<EstimateRunTimeData> result;

    if (IsGenericTritonFusion(*instr)) {
      result = indexing_model_.EstimateRunTimeForTriton(instr);
    } else {
      result = model_.EstimateRunTimeForInstruction(instr, cost_analysis);
    }

    // In case multiple threads race to compute estimates for the same
    // instruction, always return the first result.
    if (result.ok()) {
      absl::MutexLock lock(cache_mutex_);
      if (auto cached = cache_.Get(*instr)) {
        return *cached;
      }
      cache_.Set(*instr, *result);
    }
    return result;
  }
}

absl::StatusOr<CombinedGpuPerformanceModel::RunTimes>
CombinedGpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    absl::Span<const HloInstruction* const> fused_consumers) {
  ASSIGN_OR_RETURN(EstimateRunTimeData producer_runtime,
                   EstimateRunTimeForInstruction(producer, cost_analysis));

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (auto fused_consumer : fused_consumers) {
    ASSIGN_OR_RETURN(
        EstimateRunTimeData consumer_runtime,
        EstimateRunTimeForInstruction(fused_consumer, cost_analysis));

    time_unfused += consumer_runtime.exec_time;

    time_fused +=
        EstimateRunTimeForFusion(producer, fused_consumer, producer_runtime,
                                 consumer_runtime, cost_analysis,
                                 /*producer_writes_side_output=*/false);
  }

  return RunTimes{time_unfused, time_fused};
}

absl::Duration CombinedGpuPerformanceModel::EstimateRunTimeForFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis, bool producer_writes_side_output) {
  VLOG(8) << "EstimateRunTimeForFusion, producer: " << producer->name()
          << " consumer: " << consumer->name();

  {
    absl::MutexLock lock(cache_mutex_);
    if (auto cached = cache_.Get(*producer, *consumer)) {
      return *cached;
    }
  }

  absl::Duration result = EstimateRunTimeForFusionUncached(
      producer, consumer, producer_runtime, consumer_runtime, cost_analysis,
      producer_writes_side_output);

  // In case multiple threads race to compute estimates for the same
  // instruction, always return the first result.
  absl::MutexLock lock(cache_mutex_);
  if (auto cached = cache_.Get(*producer, *consumer)) {
    return *cached;
  }
  cache_.Set(*producer, *consumer, result);
  return result;
}

absl::Duration CombinedGpuPerformanceModel::EstimateRunTimeForFusionUncached(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis, bool producer_writes_side_output) {
  if (producer_runtime.IsInfinite() || consumer_runtime.IsInfinite()) {
    return absl::InfiniteDuration();
  }

  float utilization_by_this_consumer = 0;
  for (int64_t i = 0; i < consumer->operand_count(); ++i) {
    if (consumer->operand(i) == producer ||
        (consumer->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
         consumer->operand(i)->operand(0) == producer)) {
      utilization_by_this_consumer +=
          cost_analysis->operand_utilization(*consumer, i);
    }
  }

  const auto& fusion_analysis =
      fusion_analysis_cache_.Get(*producer, *consumer);

  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis, &mlir_context_);

  int64_t flops = producer_runtime.flops * utilization_by_this_consumer +
                  consumer_runtime.flops;

  absl::Duration compute_time =
      ComputeTime(device_info_, flops, launch_dimensions.num_blocks(),
                  launch_dimensions.num_threads_per_block());

  auto fusion_operands = fusion_analysis.fusion().GetParameters();
  CoalescingAnalysis coalescing_analysis = CoalescingAnalysis::Create(
      producer, consumer, fusion_operands, fusion_analysis);

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const auto* operand : fusion_operands) {
    int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());

    int64_t n_bytes_total = GetSharedOperandBytesAccessed(
        cost_analysis, producer, consumer, operand);
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    bytes_read += n_bytes_total;

    bool coalesced = coalescing_analysis.IsReadCoalesced(operand);
    PrimitiveType element_type = operand->shape().element_type();

    VLogOperandRead(operand, n_bytes_total, n_bytes_net, coalesced);

    read_time += ReadTimeWithDRAMHeuristic(
        device_info_, launch_dimensions.num_blocks(), n_bytes_net,
        n_bytes_total, operand->shape().element_type(),
        GetCoalescingUtilizationRate(element_type, device_info_, coalesced));
  }

  int64_t bytes_written = consumer_runtime.bytes_written;
  absl::Duration write_time = consumer_runtime.write_time;

  // Fusing the producer with the consumer fusion will result in a multi-output
  // fusion that writes output of the producer to the main memory. Add producer
  // output to the total memory write time.
  if (producer_writes_side_output) {
    bytes_written += producer_runtime.bytes_written;
    write_time += producer_runtime.write_time;
  }

  auto exec_time =
      CombineComputeAndMemoryAccessTime(compute_time, read_time + write_time);

  VLOG(3) << "Runtime data for producer-consumer fusion:\n"
          << " producer: " << producer->name() << "\n"
          << " consumer: " << consumer->name() << "\n"
          << launch_dimensions.ToString() << "\n"
          << EstimateRunTimeData{flops,     bytes_read, bytes_written,
                                 read_time, write_time, compute_time,
                                 exec_time}
                 .ToString();

  return exec_time;
}

absl::StatusOr<CombinedGpuPerformanceModel::RunTimes>
CombinedGpuPerformanceModel::EstimateRunTimesForMultiOutput(
    const HloInstruction* producer, const HloInstruction* consumer,
    const GpuHloCostAnalysis* cost_analysis) {
  ASSIGN_OR_RETURN(EstimateRunTimeData producer_runtime,
                   EstimateRunTimeForInstruction(producer, cost_analysis));
  ASSIGN_OR_RETURN(EstimateRunTimeData consumer_runtime,
                   EstimateRunTimeForInstruction(consumer, cost_analysis));

  absl::Duration time_unfused = 2 * kKernelLaunchOverhead +
                                producer_runtime.exec_time +
                                consumer_runtime.exec_time;

  absl::Duration time_fused =
      kKernelLaunchOverhead +
      EstimateRunTimeForFusion(producer, consumer, producer_runtime,
                               consumer_runtime, cost_analysis,
                               /*producer_writes_side_output=*/true);

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }
  return RunTimes{time_unfused, time_fused};
}

void CombinedGpuPerformanceModel::Invalidate(
    const HloInstruction& instruction) {
  absl::MutexLock lock(cache_mutex_);
  cache_.Invalidate(instruction);
}

absl::StatusOr<TiledRunTimeDataOrError>
CombinedGpuPerformanceModel::TryFindBestTilingForFusion(
    const HloFusionAdaptor& fusion_adaptor) {
  return indexing_model_.TryFindBestTilingForFusion(fusion_adaptor);
}

}  // namespace gpu
}  // namespace xla

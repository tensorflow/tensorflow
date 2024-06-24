/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_performance_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

/*static*/ EstimateRunTimeData
GpuPerformanceModel::EstimateRunTimeForInstruction(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config) {
  VLOG(8) << "EstimateRunTimeForInstruction: " << instr->name();
  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  int64_t flops = cost_analysis->flop_count(*instr);
  int64_t bytes_written = cost_analysis->output_bytes_accessed(*instr);

  // Use the analysis cache if present.
  // TODO(jreiffers): Remove this once all callers use a cache.
  std::optional<HloFusionAnalysis> local_analysis;
  if (!config.fusion_analysis_cache) {
    local_analysis = AnalyzeFusion(*instr, *cost_analysis->device_info_);
  }
  const auto& fusion_analysis = config.fusion_analysis_cache
                                    ? config.fusion_analysis_cache->Get(*instr)
                                    : local_analysis.value();
  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis);
  int64_t num_blocks = launch_dimensions.num_blocks();

  absl::Duration compute_time =
      ComputeTime(*device_info, flops, num_blocks,
                  launch_dimensions.num_threads_per_block());

  CoalescingAnalysis coalescing_analysis(instr, instr->operands(),
                                         fusion_analysis);

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const auto [operand_id, operand] : llvm::enumerate(instr->operands())) {
    int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());
    int64_t n_bytes_total =
        GetOperandBytesAccessed(cost_analysis, instr, operand);
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    bytes_read += n_bytes_total;

    bool coalesced = coalescing_analysis.IsReadCoalesced(operand);

    VLogOperandRead(operand, n_bytes_total, n_bytes_net, coalesced);

    read_time += ReadTimeWithDRAMHeuristic(
        *device_info, num_blocks, n_bytes_net, n_bytes_total,
        operand->shape().element_type(), coalesced);
  }

  absl::Duration write_time = WriteTime(*device_info, bytes_written);
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, read_time + write_time, config);

  EstimateRunTimeData runtime_data = {flops,     bytes_read, bytes_written,
                                      read_time, write_time, compute_time,
                                      exec_time};
  VLOG(3) << "Runtime data for HLO: " << instr->name() << "\n"
          << launch_dimensions.ToString() << "\n"
          << runtime_data.ToString();
  return runtime_data;
}

/*static*/ EstimateRunTimeData
GpuPerformanceModel::EstimateRunTimeForInstructionCached(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config) {
  if (config.gpu_performance_model_cache) {
    if (auto cached_result = config.gpu_performance_model_cache->Get(*instr)) {
      return *cached_result;
    }
  }

  auto runtime_data =
      EstimateRunTimeForInstruction(instr, cost_analysis, config);

  if (config.gpu_performance_model_cache) {
    config.gpu_performance_model_cache->Set(*instr, runtime_data);
  }

  return runtime_data;
}

/*static*/
absl::Duration GpuPerformanceModel::EstimateUnfusedExecTime(
    const HloInstruction* producer, const EstimateRunTimeData& producer_runtime,
    const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config,
    absl::Span<const HloInstruction* const> fused_consumers) {
  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  for (const HloInstruction* fused_consumer : fused_consumers) {
    VLOG(8) << "Unfused consumer: " << fused_consumer->name();
    float utilization_by_this_consumer =
        GetOperandUtilization(cost_analysis, fused_consumer, producer);

    // Use the analysis cache if present.
    // TODO(jreiffers): Remove this once all callers use a cache.
    std::optional<HloFusionAnalysis> local_analysis;
    if (!config.fusion_analysis_cache) {
      local_analysis = AnalyzeFusion(*fused_consumer, *device_info);
    }
    const auto& analysis_unfused =
        config.fusion_analysis_cache
            ? config.fusion_analysis_cache->Get(*fused_consumer)
            : local_analysis.value();

    LaunchDimensions launch_dimensions_unfused =
        EstimateFusionLaunchDimensions(analysis_unfused);

    int64_t n_bytes_total = std::llround(producer_runtime.bytes_written *
                                         utilization_by_this_consumer);
    int64_t n_bytes_net =
        std::min(producer_runtime.bytes_written, n_bytes_total);

    auto read_time_unfused =
        ReadTime(*device_info, launch_dimensions_unfused.num_blocks(),
                 n_bytes_net, n_bytes_total);

    VLOG(10) << "  Read time unfused: " << read_time_unfused;
    time_unfused += read_time_unfused;
  }

  return time_unfused;
}

/*static*/ absl::Duration GpuPerformanceModel::EstimateRunTimeForFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config) {
  VLOG(8) << "EstimateRunTimeForFusion, producer: " << producer->name()
          << " consumer: " << consumer->name();
  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  float utilization_by_this_consumer = 0;
  for (int64_t i = 0; i < consumer->operand_count(); ++i) {
    if (consumer->operand(i) == producer ||
        (consumer->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
         consumer->operand(i)->operand(0) == producer)) {
      utilization_by_this_consumer +=
          cost_analysis->operand_utilization(*consumer, i);
    }
  }

  std::optional<HloFusionAnalysis> local_analysis_fused;
  if (!config.fusion_analysis_cache) {
    local_analysis_fused =
        AnalyzeProducerConsumerFusion(*producer, *consumer, *device_info);
  }
  const auto& fusion_analysis =
      config.fusion_analysis_cache
          ? config.fusion_analysis_cache->Get(*producer, *consumer)
          : local_analysis_fused.value();

  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis);

  int64_t flops = producer_runtime.flops * utilization_by_this_consumer +
                  consumer_runtime.flops;

  absl::Duration compute_time =
      ComputeTime(*device_info, flops, launch_dimensions.num_blocks(),
                  launch_dimensions.num_threads_per_block());

  auto fusion_operands = fusion_analysis.fusion().GetParameters();
  CoalescingAnalysis coalescing_analysis(producer, consumer, fusion_operands,
                                         fusion_analysis);

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const auto* operand : fusion_operands) {
    int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());

    int64_t n_bytes_total = GetSharedOperandBytesAccessed(
        cost_analysis, producer, consumer, operand);
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    bytes_read += n_bytes_total;

    bool coalesced = coalescing_analysis.IsReadCoalesced(operand);

    VLogOperandRead(operand, n_bytes_total, n_bytes_net, coalesced);

    read_time += ReadTimeWithDRAMHeuristic(
        *device_info, launch_dimensions.num_blocks(), n_bytes_net,
        n_bytes_total, operand->shape().element_type(), coalesced);
  }

  auto exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, read_time + consumer_runtime.write_time, config);

  VLOG(3) << "Runtime data for producer-consumer fusion:\n"
          << " producer: " << producer->name() << "\n"
          << " consumer: " << consumer->name() << "\n"
          << launch_dimensions.ToString() << "\n"
          << EstimateRunTimeData{flops,
                                 bytes_read,
                                 consumer_runtime.bytes_written,
                                 read_time,
                                 consumer_runtime.write_time,
                                 compute_time,
                                 exec_time}
                 .ToString();

  return exec_time;
}

/*static*/
absl::Duration GpuPerformanceModel::EstimateRunTimeForFusionCached(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config) {
  if (config.gpu_performance_model_cache) {
    if (auto fusion_runtime =
            config.gpu_performance_model_cache->Get(*producer, *consumer)) {
      return *fusion_runtime;
    }
  }

  auto fusion_runtime =
      EstimateRunTimeForFusion(producer, consumer, producer_runtime,
                               consumer_runtime, cost_analysis, config);

  if (config.gpu_performance_model_cache) {
    config.gpu_performance_model_cache->Set(*producer, *consumer,
                                            fusion_runtime);
  }
  return fusion_runtime;
}

/*static*/
absl::Duration GpuPerformanceModel::EstimateFusedExecTime(
    const HloInstruction* producer, const EstimateRunTimeData& producer_runtime,
    const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config,
    absl::Span<const HloInstruction* const> fused_consumers,
    bool multi_output) {
  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  absl::Duration exec_time_fused =
      kKernelLaunchOverhead * fused_consumers.size();
  for (auto [idx, fused_consumer] : llvm::enumerate(fused_consumers)) {
    VLOG(8) << "Fused consumer: " << fused_consumer->name();

    float utilization_by_this_consumer = cost_analysis->operand_utilization(
        *fused_consumer, fused_consumer->operand_index(producer));

    std::optional<HloFusionAnalysis> local_analysis_fused;
    if (!config.fusion_analysis_cache) {
      local_analysis_fused = AnalyzeProducerConsumerFusion(
          *producer, *fused_consumer, *device_info);
    }
    const auto& analysis_fused =
        config.fusion_analysis_cache
            ? config.fusion_analysis_cache->Get(*producer, *fused_consumer)
            : local_analysis_fused.value();

    LaunchDimensions launch_dimensions_fused =
        EstimateFusionLaunchDimensions(analysis_fused);

    absl::Duration compute_time_by_this_consumer = ComputeTime(
        *device_info, producer_runtime.flops * utilization_by_this_consumer,
        launch_dimensions_fused.num_blocks(),
        launch_dimensions_fused.num_threads_per_block());

    // Here, we assume that the read is distributed over all the threads in the
    // launch grid. Usually this is the case, but not always: for example, a
    // reduce -> broadcast -> elementwise fusion will recompute the reduce. We
    // don't currently have an analysis that is able to detect these cases.
    absl::Duration input_access_time_by_this_consumer = ProducerInputAccessTime(
        cost_analysis, *device_info, launch_dimensions_fused.num_blocks(),
        producer, analysis_fused, config, fused_consumer);
    VLOG(10) << "  Compute time by consumer: " << compute_time_by_this_consumer;
    VLOG(10) << "  Input access time by consumer: "
             << input_access_time_by_this_consumer;

    exec_time_fused += CombineComputeAndMemoryAccessTime(
        compute_time_by_this_consumer, input_access_time_by_this_consumer,
        config);
  }

  // Multi-output fusion still writes the initial output of the producer.
  // For now assume that the producer's output does not need to be recomputed.
  if (multi_output) {
    exec_time_fused += producer_runtime.write_time;
  }

  return exec_time_fused;
}

/*static*/
GpuPerformanceModel::RunTimes
GpuPerformanceModel::EstimateRunTimesForPriorityFusion(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config,
    absl::Span<const HloInstruction* const> fused_consumers,
    bool multi_output) {
  EstimateRunTimeData producer_runtime =
      EstimateRunTimeForInstructionCached(producer, cost_analysis, config);

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (auto fused_consumer : fused_consumers) {
    VLOG(8) << "Fused consumer: " << fused_consumer->name();

    EstimateRunTimeData consumer_runtime = EstimateRunTimeForInstructionCached(
        fused_consumer, cost_analysis, config);

    time_unfused += consumer_runtime.exec_time;

    time_fused += EstimateRunTimeForFusionCached(
        producer, fused_consumer, producer_runtime, consumer_runtime,
        cost_analysis, config);
  }

  // Multi-output fusion still writes the initial output of the producer.
  // For now assume that the producer's output does not need to be recomputed.
  if (multi_output) {
    time_fused += producer_runtime.write_time;
  }

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Consumer count: " << fused_consumers.size();
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

/*static*/
GpuPerformanceModel::RunTimes GpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config,
    absl::Span<const HloInstruction* const> fused_consumers,
    bool multi_output) {
  VLOG(8) << "Producer: " << producer->name();
  if (producer->opcode() == HloOpcode::kFusion) {
    VLOG(10) << producer->fused_instructions_computation()->ToString();
  }

  EstimateRunTimeData producer_runtime =
      EstimateRunTimeForInstructionCached(producer, cost_analysis, config);

  absl::Duration time_unfused = EstimateUnfusedExecTime(
      producer, producer_runtime, cost_analysis, config, fused_consumers);

  absl::Duration time_fused =
      EstimateFusedExecTime(producer, producer_runtime, cost_analysis, config,
                            fused_consumers, multi_output);

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Consumer count: " << fused_consumers.size();
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

/*static*/
void GpuPerformanceModel::RecordEstimatedRunTime(
    HloInstruction* instruction, const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config) {
  DCHECK(Cast<const HloFusionInstruction>(instruction)) << "expected fusion";
  DCHECK(cost_analysis != nullptr) << "expected cost analysis";

  EstimateRunTimeData data =
      EstimateRunTimeForInstructionCached(instruction, cost_analysis, config);
  double cycles = absl::ToDoubleNanoseconds(data.exec_time) *
                  cost_analysis->device_info_->clock_rate_ghz();

  auto gpu_config = instruction->backend_config<GpuBackendConfig>();
  TF_CHECK_OK(gpu_config.status()) << instruction->ToString();
  auto reification_cost =
      gpu_config->mutable_fusion_backend_config()->mutable_reification_cost();
  reification_cost->set_end_to_end_cycles(cycles);
  reification_cost->set_compute_time_us(
      absl::ToDoubleMicroseconds(data.compute_time));
  reification_cost->set_memory_access_time_us(
      absl::ToDoubleMicroseconds(data.read_time + data.write_time));
  reification_cost->set_exec_time_us(
      absl::ToDoubleMicroseconds(data.exec_time));
  TF_CHECK_OK(instruction->set_backend_config(*gpu_config));

  VLOG(8) << "RecordEstimatedRunTime: " << instruction->ToString();
}

}  // namespace gpu
}  // namespace xla

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

#include <cstdint>
#include <memory>
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
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

GpuPerformanceModel::GpuPerformanceModel(
    const se::DeviceDescription& device_info,
    HloFusionAnalysisCache& fusion_analysis_cache,
    GpuPerformanceModelCache& gpu_performance_model_cache)
    : device_info_(device_info),
      fusion_analysis_cache_(fusion_analysis_cache),
      gpu_performance_model_cache_(gpu_performance_model_cache) {};

EstimateRunTimeData GpuPerformanceModel::EstimateRunTimeForInstructionImpl(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  VLOG(8) << "EstimateRunTimeForInstruction: " << instr->name();

  int64_t flops = cost_analysis->flop_count(*instr);
  int64_t bytes_written = cost_analysis->output_bytes_accessed(*instr);

  const auto& fusion_analysis = fusion_analysis_cache_.Get(*instr);
  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis);
  int64_t num_blocks = launch_dimensions.num_blocks();

  absl::Duration compute_time =
      ComputeTime(device_info_, flops, num_blocks,
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
    PrimitiveType element_type = operand->shape().element_type();

    VLogOperandRead(operand, n_bytes_total, n_bytes_net, coalesced);

    read_time += ReadTimeWithDRAMHeuristic(
        device_info_, num_blocks, n_bytes_net, n_bytes_total,
        operand->shape().element_type(),
        GetCoalescingUtilizationRate(element_type, device_info_, coalesced));
  }

  absl::Duration write_time = WriteTime(device_info_, bytes_written);
  absl::Duration exec_time =
      CombineComputeAndMemoryAccessTime(compute_time, read_time + write_time);

  EstimateRunTimeData runtime_data = {flops,     bytes_read, bytes_written,
                                      read_time, write_time, compute_time,
                                      exec_time};
  VLOG(3) << "Runtime data for HLO: " << instr->name() << "\n"
          << launch_dimensions.ToString() << "\n"
          << runtime_data.ToString();
  return runtime_data;
}

EstimateRunTimeData GpuPerformanceModel::EstimateRunTimeForInstruction(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  if (auto cached_result_opt = gpu_performance_model_cache_.Get(*instr)) {
    return *cached_result_opt;
  }

  auto runtime_data = EstimateRunTimeForInstructionImpl(instr, cost_analysis);

  gpu_performance_model_cache_.Set(*instr, runtime_data);

  return runtime_data;
}

absl::Duration GpuPerformanceModel::EstimateRunTimeForFusionImpl(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis, bool producer_writes_side_output) {
  VLOG(8) << "EstimateRunTimeForFusion, producer: " << producer->name()
          << " consumer: " << consumer->name();

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
      EstimateFusionLaunchDimensions(fusion_analysis);

  int64_t flops = producer_runtime.flops * utilization_by_this_consumer +
                  consumer_runtime.flops;

  absl::Duration compute_time =
      ComputeTime(device_info_, flops, launch_dimensions.num_blocks(),
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

absl::Duration GpuPerformanceModel::EstimateRunTimeForFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis, bool producer_writes_side_output) {
  if (auto fusion_runtime_opt =
          gpu_performance_model_cache_.Get(*producer, *consumer)) {
    return *fusion_runtime_opt;
  }

  auto fusion_runtime = EstimateRunTimeForFusionImpl(
      producer, consumer, producer_runtime, consumer_runtime, cost_analysis,
      producer_writes_side_output);

  gpu_performance_model_cache_.Set(*producer, *consumer, fusion_runtime);
  return fusion_runtime;
}

GpuPerformanceModel::RunTimes GpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    absl::Span<const HloInstruction* const> fused_consumers) {
  auto cache_result = gpu_performance_model_cache_.Get(*producer);
  CHECK(cache_result.has_value());
  EstimateRunTimeData producer_runtime = *cache_result;

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (auto fused_consumer : fused_consumers) {
    VLOG(8) << "Fused consumer: " << fused_consumer->name();

    auto cache_result = gpu_performance_model_cache_.Get(*fused_consumer);
    CHECK(cache_result.has_value());
    EstimateRunTimeData consumer_runtime = *cache_result;

    time_unfused += consumer_runtime.exec_time;

    time_fused +=
        EstimateRunTimeForFusion(producer, fused_consumer, producer_runtime,
                                 consumer_runtime, cost_analysis);
  }

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Consumer count: " << fused_consumers.size();
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

GpuPerformanceModel::RunTimes
GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const GpuHloCostAnalysis* cost_analysis) {
  EstimateRunTimeData producer_runtime =
      EstimateRunTimeForInstruction(producer, cost_analysis);
  EstimateRunTimeData consumer_runtime =
      EstimateRunTimeForInstruction(consumer, cost_analysis);

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

  return {time_unfused, time_fused};
}

void GpuPerformanceModel::RecordEstimatedRunTime(
    HloInstruction* instruction, const GpuHloCostAnalysis* cost_analysis) {
  DCHECK(Cast<const HloFusionInstruction>(instruction)) << "expected fusion";
  DCHECK(cost_analysis != nullptr) << "expected cost analysis";

  EstimateRunTimeData data =
      EstimateRunTimeForInstruction(instruction, cost_analysis);
  double cycles =
      absl::ToDoubleNanoseconds(data.exec_time) * device_info_.clock_rate_ghz();

  auto gpu_config = instruction->backend_config<GpuBackendConfig>();
  TF_CHECK_OK(gpu_config.status()) << instruction->ToString();
  auto reification_cost = gpu_config->add_reification_cost();
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

GpuPerformanceModelOwning::GpuPerformanceModelOwning(
    const se::DeviceDescription& device_info)
    : fusion_analysis_cache_(device_info),
      gpu_performance_model_(std::make_unique<GpuPerformanceModel>(
          device_info, fusion_analysis_cache_, gpu_performance_model_cache_)) {
      };

void GpuPerformanceModelOwning::RecordEstimatedRunTime(
    HloInstruction* instruction,
    const GpuHloCostAnalysis* cost_analysis) const {
  gpu_performance_model_->RecordEstimatedRunTime(instruction, cost_analysis);
}

EstimateRunTimeData GpuPerformanceModelOwning::EstimateRunTimeForInstruction(
    const HloInstruction* instr,
    const GpuHloCostAnalysis* cost_analysis) const {
  return gpu_performance_model_->EstimateRunTimeForInstruction(instr,
                                                               cost_analysis);
}

GpuPerformanceModel::RunTimes
GpuPerformanceModelOwning::EstimateRunTimesForMultiOutputFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const GpuHloCostAnalysis* cost_analysis) const {
  return gpu_performance_model_->EstimateRunTimesForMultiOutputFusion(
      producer, consumer, cost_analysis);
}

}  // namespace gpu
}  // namespace xla

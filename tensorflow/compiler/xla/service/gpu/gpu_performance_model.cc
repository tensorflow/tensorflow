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
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_fusion_analysis.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace {

// Estimated values in the absence of easy ways to query them.
static constexpr absl::Duration kKernelLaunchOverhead = absl::Microseconds(1);
static constexpr float kL2CacheSpeedup = 2.5;
static constexpr float kL1CacheSpeedup = 8;
// A very conservative estimate. L1 size varies because it can be dynamically
// configured as shared memory; there is no easy way to query its actual size;
// also we do not count what occupies cache, but rather claim that what is
// much smaller than the cache size will likely stay in it.
// For reference, it can be up to 256 kB per SM on RTX A6000.
static constexpr float kL1CacheSizePerSM = 2 * 1024;

// Returns whether a fusion uses the parameter at the given index elementwise
// from its root.
bool FusionUsesParameterElementwiseFromRoot(
    const HloInstruction* fusion, int parameter_index,
    const GpuHloCostAnalysis* cost_analysis) {
  return cost_analysis->CommonElementwiseUtilization(
             fusion->fused_parameter(parameter_index),
             fusion->fused_expression_root()) == 1.f;
}

// Estimate read time of n_bytes_total bytes from global memory on a
// given GPU. Account for L1 / L2 cache speedup if the input's nominal size
// n_bytes_net is small.
absl::Duration ReadTime(const GpuDeviceInfo& gpu_device_info,
                        int64_t n_bytes_net, int64_t n_bytes_total) {
  float bw = gpu_device_info.memory_bandwidth;
  if (n_bytes_net < gpu_device_info.l2_cache_size) {
    bw *= kL2CacheSpeedup;
    if (n_bytes_net < kL1CacheSizePerSM * gpu_device_info.core_count) {
      bw *= kL1CacheSpeedup;
    }
  }
  return absl::Seconds(n_bytes_total / bw);
}

// Tells input access time of the producer alone if fused_consumer
// is not specified. Otherwise estimates the access time to producer's
// inputs as if it is fused into the consumer.
absl::Duration ProducerInputAccessTime(
    const GpuHloCostAnalysis* cost_analysis,
    const GpuDeviceInfo& gpu_device_info, const HloInstruction* producer,
    const HloInstruction* fused_consumer = nullptr) {
  absl::Duration ret = absl::ZeroDuration();
  float producer_output_utilization = 1.f;
  ConstHloInstructionSet consumer_operands;
  if (fused_consumer) {
    producer_output_utilization = cost_analysis->operand_utilization(
        *fused_consumer, fused_consumer->operand_index(producer));
    for (const HloInstruction* op : fused_consumer->operands()) {
      consumer_operands.insert(op);
    }
  }
  for (int i = 0; i < producer->operand_count(); ++i) {
    int64_t p_size_accessed =
        cost_analysis->operand_bytes_accessed(*producer, i);
    float operand_utilization =
        cost_analysis->operand_utilization(*producer, i);
    int64_t p_size_net =
        (operand_utilization == 0)
            ? 0
            : static_cast<float>(p_size_accessed) / operand_utilization;
    // Look for common operands of producer and consumer that are accessed
    // more efficiently on merge:
    // 1) Producer has to use the common operand elementwise from its root if
    //    it is a fusion or just be an elementwise instruction.
    // 2) Consumer has to have common elementwise roots for the producer
    //    and the common operand if it is a fusion or just be an elementwise
    //    instruction.
    float common_utilization = 0;
    if (consumer_operands.count(producer->operand(i)) &&
        (producer->IsElementwise() ||
         (producer->opcode() == HloOpcode::kFusion &&
          FusionUsesParameterElementwiseFromRoot(producer, i,
                                                 cost_analysis)))) {
      if (fused_consumer->opcode() == HloOpcode::kFusion) {
        int64_t consumer_idx_of_common_operand =
            fused_consumer->operand_index(producer->operand(i));
        int64_t consumer_idx_of_producer =
            fused_consumer->operand_index(producer);
        common_utilization = cost_analysis->CommonElementwiseUtilization(
            fused_consumer->fused_parameter(consumer_idx_of_common_operand),
            fused_consumer->fused_parameter(consumer_idx_of_producer));
      } else {
        if (fused_consumer->IsElementwise()) {
          common_utilization = 1.f;
        }
      }
    }
    CHECK_LE(common_utilization, producer_output_utilization);
    ret += ReadTime(
        gpu_device_info, std::min(p_size_net, p_size_accessed),
        p_size_accessed * (producer_output_utilization - common_utilization));
  }
  return ret;
}

// Use HloFusionAnalysis for computing the actual number of threads that the
// IR emitter will use. Return std::nullopt if this data is not available.
std::optional<int64_t> EstimateThreadCount(
    const HloInstruction* instr, const GpuDeviceInfo& gpu_device_info,
    std::optional<se::CudaComputeCapability> cc) {
  auto fusion = DynCast<const HloFusionInstruction>(instr);
  if (fusion != nullptr && cc.has_value()) {
    auto analysis =
        HloFusionAnalysis::Create(fusion, &gpu_device_info, cc.value());
    if (analysis.ok()) {
      auto launch_dimensions = analysis->GetLaunchDimensions();
      if (launch_dimensions.ok()) {
        return launch_dimensions->launch_bound();
      }
    }
  }
  return std::nullopt;
}

absl::Duration ComputeTime(const GpuDeviceInfo& gpu_device_info,
                           int64_t n_flops, int64_t n_threads) {
  int fpu_count = gpu_device_info.core_count * gpu_device_info.fpus_per_core;
  float n_threads_active = fmin(n_threads, fpu_count);
  float flop_per_second_per_fpu = 2 * 1e9 * gpu_device_info.clock_rate_ghz;
  float flop_per_second_effective = flop_per_second_per_fpu * n_threads_active;
  return absl::Seconds(n_flops / flop_per_second_effective);
}

struct EstimateRunTimeData {
  int64_t flops;
  float bytes_written;
  float elements_out;
  absl::Duration write_time;
  absl::Duration exec_time;
};

EstimateRunTimeData EstimateRunTimeImpl(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  int64_t flops = cost_analysis->flop_count(*instr);
  float bytes_written = cost_analysis->output_bytes_accessed(*instr);
  float bytes_read = cost_analysis->bytes_accessed(*instr) - bytes_written;
  float elements_out = ShapeUtil::ElementsInRecursive(instr->shape());

  absl::Duration compute_time =
      ComputeTime(*cost_analysis->device_info_, flops, elements_out);
  absl::Duration read_time = ProducerInputAccessTime(
      cost_analysis, *cost_analysis->device_info_, instr);
  absl::Duration write_time = absl::Seconds(
      bytes_written / cost_analysis->device_info_->memory_bandwidth);
  absl::Duration exec_time = std::max(compute_time, read_time + write_time);

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "FLOPs: " << flops;
    LOG(INFO) << "Bytes read: " << bytes_read;
    LOG(INFO) << "Bytes written: " << bytes_written;
    LOG(INFO) << "Elements out: " << elements_out;
    LOG(INFO) << "Compute time: " << compute_time;
    LOG(INFO) << "Input read time: " << read_time;
    LOG(INFO) << "Output write time: " << write_time;
  }

  return {flops, bytes_written, elements_out, write_time, exec_time};
}

}  // namespace

GpuPerformanceModel::RunTimes GpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    std::optional<se::CudaComputeCapability> cc,
    std::vector<HloInstruction*> fused_users, bool multi_output) {
  VLOG(8) << "Producer: " << producer->name();
  if (producer->opcode() == HloOpcode::kFusion) {
    VLOG(10) << producer->fused_instructions_computation()->ToString();
  }

  EstimateRunTimeData producer_data =
      EstimateRunTimeImpl(producer, cost_analysis);

  int64_t fused_consumer_count = fused_users.size();
  float total_producer_utilization = 0;

  absl::Duration exec_time_fused = absl::ZeroDuration();
  absl::Duration producer_output_read_time_unfused = absl::ZeroDuration();
  for (const HloInstruction* u : fused_users) {
    float utilization_by_this_consumer =
        cost_analysis->operand_utilization(*u, u->operand_index(producer));
    total_producer_utilization += utilization_by_this_consumer;

    auto thread_count =
        EstimateThreadCount(u, *cost_analysis->device_info_, cc);
    int64_t upper_bound =
        producer_data.elements_out * utilization_by_this_consumer;
    absl::Duration compute_time_by_this_consumer = ComputeTime(
        *cost_analysis->device_info_,
        producer_data.flops * utilization_by_this_consumer,
        thread_count.has_value() ? std::min(*thread_count, upper_bound)
                                 : upper_bound);
    exec_time_fused +=
        std::max(compute_time_by_this_consumer,
                 ProducerInputAccessTime(
                     cost_analysis, *cost_analysis->device_info_, producer, u));
    producer_output_read_time_unfused += ReadTime(
        *cost_analysis->device_info_,
        std::min(producer_data.bytes_written,
                 producer_data.bytes_written * utilization_by_this_consumer),
        producer_data.bytes_written * utilization_by_this_consumer);
  }

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumer_count + 1) +
      producer_data.exec_time + producer_output_read_time_unfused;

  absl::Duration time_fused =
      kKernelLaunchOverhead * fused_consumer_count + exec_time_fused;
  // Multi-output fusion still writes the initial output of the producer.
  // For now assume that the producer's output does not need to be recomputed.
  if (multi_output) {
    time_fused += producer_data.write_time;
  }

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Consumer count: " << fused_consumer_count;
    LOG(INFO) << "Utilization of producer output: "
              << total_producer_utilization;
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

void GpuPerformanceModel::RecordEstimatedRunTime(
    HloInstruction* instruction, const GpuHloCostAnalysis* cost_analysis) {
  DCHECK(Cast<const HloFusionInstruction>(instruction)) << "expected fusion";
  DCHECK(cost_analysis != nullptr) << "expected cost analysis";

  EstimateRunTimeData data = EstimateRunTimeImpl(instruction, cost_analysis);
  double cycles = absl::ToDoubleNanoseconds(data.exec_time) *
                  cost_analysis->device_info_->clock_rate_ghz;

  auto backend_config = instruction->backend_config<FusionBackendConfig>();
  TF_CHECK_OK(backend_config.status()) << instruction->ToString();
  backend_config->mutable_reification_cost()->set_end_to_end_cycles(cycles);
  TF_CHECK_OK(instruction->set_backend_config(*backend_config));

  VLOG(8) << "RecordEstimatedRunTime: " << instruction->ToString();
}

}  // namespace gpu
}  // namespace xla

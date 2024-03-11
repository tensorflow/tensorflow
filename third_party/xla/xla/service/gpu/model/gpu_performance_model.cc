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
#include <cstdlib>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
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
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/status.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/nvml/include/nvml.h"
#endif  // GOOGLE_CUDA
namespace xla {
namespace gpu {

namespace {

int64_t GetNcclMaxNumChannels(
    GpuPerformanceWithCollectiveModel::CollectiveAlgo algorithm) {
  int64_t max_nchannels = 0;
  switch (algorithm) {
    // Tree and Ring algos share the same max channel number.
    case GpuPerformanceWithCollectiveModel::RING:
    case GpuPerformanceWithCollectiveModel::TREE:
      max_nchannels = GpuPerformanceWithCollectiveModel::kMaxNumChannelsRing;
      break;
  }
  const char* env = std::getenv("NCCL_MAX_NCHANNELS");
  if (env != nullptr) {
    int64_t max_nchannels_from_env;
    if (absl::SimpleAtoi(env, &max_nchannels_from_env)) {
      max_nchannels = std::min(max_nchannels_from_env, max_nchannels);
    }
  }
  return max_nchannels;
}

int64_t GetMinNumberOfChannels(
    GpuPerformanceWithCollectiveModel::CollectiveAlgo algorithm) {
  int64_t min_nchannels = 0;
  switch (algorithm) {
    // Tree and Ring algos share the same min channel number.
    case GpuPerformanceWithCollectiveModel::RING:
    case GpuPerformanceWithCollectiveModel::TREE:
      min_nchannels = 1;
      break;
  }
  const char* env = std::getenv("NCCL_MIN_NCHANNELS");
  if (env != nullptr) {
    int64_t min_nchannels_from_env;
    if (absl::SimpleAtoi(env, &min_nchannels_from_env)) {
      min_nchannels = std::min(min_nchannels_from_env, min_nchannels);
    }
  }
  return min_nchannels;
}

int GetNumThreads(int warp_size, int min_num_threads, int max_num_threads,
                  int default_num_threads) {
  int threads_from_env = default_num_threads;
  const char* env = std::getenv("NCCL_NTHREADS");
  if (env != nullptr) {
    CHECK(absl::SimpleAtoi(env, &threads_from_env));
  }
  int num_threads = threads_from_env;
  if (num_threads > 0) {
    if (num_threads % warp_size != 0) {
      num_threads = max_num_threads;
    } else if (num_threads > max_num_threads) {
      num_threads = max_num_threads;
    } else if (num_threads < min_num_threads) {
      num_threads = min_num_threads;
    }
  } else {
    num_threads = default_num_threads;
  }
  return num_threads;
}

float GetMaxSysBwFromGpu(const se::CudaComputeCapability cc,
                         const double* bandwidths_table) {
  switch (cc.major) {
    case se::CudaComputeCapability::VOLTA:
      return bandwidths_table[0];
    case se::CudaComputeCapability::AMPERE:
      return bandwidths_table[1];
    case se::CudaComputeCapability::HOPPER:
      return bandwidths_table[2];
  }
  return -1;
}

}  // namespace

/*static*/ EstimateRunTimeData
GpuPerformanceModel::EstimateRunTimeForInstruction(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis,
    const GpuPerformanceModelOptions& config) {
  VLOG(8) << "EstimateRunTimeForInstruction: " << instr->name();
  const se::DeviceDescription* device_info = cost_analysis->device_info_;

  int64_t flops = cost_analysis->flop_count(*instr);
  int64_t bytes_written = cost_analysis->output_bytes_accessed(*instr);
  int64_t bytes_read = cost_analysis->bytes_accessed(*instr) - bytes_written;

  // Use the analysis cache if present.
  // TODO(jreiffers): Remove this once all callers use a cache.
  std::optional<HloFusionAnalysis> local_analysis;
  if (!config.fusion_analysis_cache) {
    local_analysis = AnalyzeFusion(*instr, *cost_analysis->device_info_);
  }
  const auto& fusion_analysis = config.fusion_analysis_cache
                                    ? config.fusion_analysis_cache->Get(*instr)
                                    : local_analysis.value();
  LaunchDimensions launch_dimensions = EstimateFusionLaunchDimensions(
      ShapeUtil::ElementsInRecursive(instr->shape()), fusion_analysis,
      *device_info);
  int64_t num_threads = launch_dimensions.launch_bound();
  int64_t num_blocks = launch_dimensions.num_blocks();

  absl::Duration compute_time = ComputeTime(*device_info, flops, num_threads);

  // TODO(jreiffers): We should be checking each operand.
  bool coalesced = IsReadCoalescedHeuristic(fusion_analysis, instr,
                                            /*consumer=*/nullptr);

  absl::Duration read_time;
  for (int i = 0; i < instr->operand_count(); ++i) {
    auto element_type = instr->operand(i)->shape().element_type();
    // Information about data read taking into account utilization.
    // If `operand_utilization` is 0, `operand_bytes_accessed` should be also 0.
    int64_t n_bytes_total = cost_analysis->operand_bytes_accessed(*instr, i);
    float operand_utilization = cost_analysis->operand_utilization(*instr, i);

    // An estimate how much data would need to fit into L1/L2 cache to speed up
    // the operand access.
    // If `operand_utilization` < 1, only a part of the full operand size should
    // be read. Otherwise, `n_bytes_total / operand_utilization` is the
    // size of the operand without reuse.
    int64_t n_bytes_net =
        std::llround(n_bytes_total / std::max(operand_utilization, 1.0f));

    read_time +=
        ReadTimeWithDRAMHeuristic(*device_info, num_blocks, n_bytes_net,
                                  n_bytes_total, element_type, coalesced);
  }

  absl::Duration write_time = WriteTime(*device_info, bytes_written);
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, read_time + write_time, config);

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "FLOPs: " << flops;
    LOG(INFO) << "Bytes read: " << bytes_read;
    LOG(INFO) << "Bytes written: " << bytes_written;
    LOG(INFO) << "Num threads: " << num_threads;
    LOG(INFO) << "Compute time: " << compute_time;
    LOG(INFO) << "Input read time: " << read_time;
    LOG(INFO) << "Output write time: " << write_time;
  }

  return {flops, bytes_written, num_threads, write_time, exec_time};
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

    LaunchDimensions launch_dimensions_unfused = EstimateFusionLaunchDimensions(
        ShapeUtil::ElementsInRecursive(fused_consumer->shape()),
        analysis_unfused, *device_info);

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

  float utilization_by_this_consumer = cost_analysis->operand_utilization(
      *consumer, consumer->operand_index(producer));

  std::optional<HloFusionAnalysis> local_analysis_fused;
  if (!config.fusion_analysis_cache) {
    local_analysis_fused =
        AnalyzeProducerConsumerFusion(*producer, *consumer, *device_info);
  }
  const auto& fusion_analysis =
      config.fusion_analysis_cache
          ? config.fusion_analysis_cache->Get(*producer, *consumer)
          : local_analysis_fused.value();

  LaunchDimensions launch_dimensions = EstimateFusionLaunchDimensions(
      producer_runtime.num_threads * utilization_by_this_consumer,
      fusion_analysis, *device_info);

  int64_t fused_flops = producer_runtime.flops * utilization_by_this_consumer +
                        consumer_runtime.flops;

  int64_t num_threads = launch_dimensions.launch_bound();
  absl::Duration compute_time =
      ComputeTime(*device_info, fused_flops, num_threads);

  absl::flat_hash_set<const HloInstruction*> fusion_operands;
  for (auto* operand : producer->operands()) {
    fusion_operands.insert(operand);
  }
  for (auto* operand : consumer->operands()) {
    if (operand != producer) {
      fusion_operands.insert(operand);
    }
  }

  absl::Duration read_time;
  for (const auto* operand : fusion_operands) {
    float operand_utilization =
        GetSharedUtilization(cost_analysis, producer, consumer, operand);

    int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());

    int64_t n_bytes_total = std::llround(operand_size * operand_utilization);
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);

    bool coalesced =
        IsReadCoalescedHeuristic(fusion_analysis, producer, consumer);

    read_time += ReadTimeWithDRAMHeuristic(
        *device_info, launch_dimensions.num_blocks(), n_bytes_net,
        n_bytes_total, operand->shape().element_type(), coalesced);
  }

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Fused FLOPs: " << fused_flops;
    LOG(INFO) << "Num threads: " << num_threads;
    LOG(INFO) << "Compute time: " << compute_time;
    LOG(INFO) << "Input read time: " << read_time;
    LOG(INFO) << "Output write time: " << consumer_runtime.write_time;
  }

  return CombineComputeAndMemoryAccessTime(
      compute_time, read_time + consumer_runtime.write_time, config);
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

    LaunchDimensions launch_dimensions_fused = EstimateFusionLaunchDimensions(
        producer_runtime.num_threads * utilization_by_this_consumer,
        analysis_fused, *device_info);

    absl::Duration compute_time_by_this_consumer = ComputeTime(
        *device_info, producer_runtime.flops * utilization_by_this_consumer,
        launch_dimensions_fused.launch_bound());

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

  int64_t fused_consumer_count = fused_consumers.size();
  float total_producer_utilization = 0;

  for (const HloInstruction* fused_consumer : fused_consumers) {
    float utilization_by_this_consumer = cost_analysis->operand_utilization(
        *fused_consumer, fused_consumer->operand_index(producer));
    total_producer_utilization += utilization_by_this_consumer;
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
  FusionBackendConfig& backend_config =
      *gpu_config->mutable_fusion_backend_config();
  backend_config.mutable_reification_cost()->set_end_to_end_cycles(cycles);
  TF_CHECK_OK(instruction->set_backend_config(*gpu_config));

  VLOG(8) << "RecordEstimatedRunTime: " << instruction->ToString();
}

// Returns NVLink bw in GB/s
/*static*/
float GpuPerformanceWithCollectiveModel::GetNvlinkBw(
    se::CudaComputeCapability compute_capability) {
  return compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER)
             ? kSm90NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE)
             ? kSm80NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::VOLTA)
             ? kSm70NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::PASCAL_)
             ? kSm60NvlinkBandwidth
             : kSm80NvlinkBandwidth;
}

/*static*/ bool GpuPerformanceWithCollectiveModel::InitNvml() {
#if GOOGLE_CUDA
  void* libhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  CHECK(libhandle != nullptr) << "Failed to open libnvidia-ml.so.1";

  struct SymbolEntry {
    void** functor;
    char const* name;
  };

  std::vector<SymbolEntry> symbols = {
      {(void**)&xla_nvmlInit, "nvmlInit_v2"},
      {(void**)&xla_nvmlShutdown, "nvmlShutdown"},
      {(void**)&xla_nvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex"},
      {(void**)&xla_nvmlDeviceGetNvLinkCapability,
       "nvmlDeviceGetNvLinkCapability"},
  };
  for (SymbolEntry se : symbols) {
    *se.functor = dlsym(libhandle, se.name);
  }
  nvmlReturn_t init_result = xla_nvmlInit();
  return init_result == NVML_SUCCESS;
#else
  return false;
#endif  // GOOGLE_CUDA
}

/*static*/ bool GpuPerformanceWithCollectiveModel::ShutdownNvml() {
#if GOOGLE_CUDA
  nvmlReturn_t shutdown_result = xla_nvmlShutdown();
  return shutdown_result == NVML_SUCCESS;
#else
  return false;
#endif  // GOOGLE_CUDA
}

/*static*/ uint32_t
GpuPerformanceWithCollectiveModel::CheckIfNvlinkSupportsP2P() {
#if GOOGLE_CUDA
  // We will use nvml library to detect nvlink capability
  // to see if it supports p2p communication.
  // We first load libnvidia-ml.so and assign symbols to function pointers
  // to avoid linking errors.
  // Then gpu 0 will be used to query for nvlink capability, note that
  // we only look at link 0 of gpu 0 since all other links are assumed
  // to have the same capability.
  CHECK(InitNvml()) << "NVML init failed.";
  nvmlDevice_t nvml_device;
  nvmlReturn_t get_device_result =
      xla_nvmlDeviceGetHandleByIndex(0, &nvml_device);
  CHECK(get_device_result == NVML_SUCCESS);

  uint32_t supported_p2p = 0;

  nvmlReturn_t nvlink_cap_result = xla_nvmlDeviceGetNvLinkCapability(
      nvml_device, /*nvlink link number*/ 0, NVML_NVLINK_CAP_P2P_SUPPORTED,
      &supported_p2p);
  CHECK(nvlink_cap_result == NVML_SUCCESS);
  CHECK(ShutdownNvml()) << "NVML shutdown failed.";
  return supported_p2p;
#else
  return 0;
#endif  // GOOGLE_CUDA
}

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeAllreduceTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  absl::Duration total_time = kNcclKernelLaunchOverhead;
  stream_executor::CudaComputeCapability compute_cap =
      gpu_device_info.cuda_compute_capability();

  int64_t size_of_speed_array = kIntraNodeSpeeds.size();
  int64_t size_of_sm90_speed_array = kIntraNodeSpeedsSm90.size();

  int num_speeds = compute_cap.major >= se::CudaComputeCapability::HOPPER
                       ? size_of_sm90_speed_array
                       : size_of_speed_array;
  const double* speeds = compute_cap.major >= se::CudaComputeCapability::HOPPER
                             ? kIntraNodeSpeedsSm90.data()
                             : kIntraNodeSpeeds.data();

  int speed_index = 0;
  float max_sys_bw =
      GetMaxSysBwFromGpu(compute_cap, kLowLatencyMaxBandwidths.data());

  CHECK_GT(max_sys_bw, 0);

  while ((speed_index < num_speeds - 1) && speeds[speed_index] > max_sys_bw) {
    speed_index++;
  }
  float bw_intra_node = speeds[speed_index];
  int64_t num_devices = cost_analysis->NumOfDevices(instr);

  int64_t min_nchannels =
      std::max(num_devices, GetMinNumberOfChannels(CollectiveAlgo::RING));
  int64_t num_channels =
      std::max(min_nchannels, GetNcclMaxNumChannels(CollectiveAlgo::RING));
  int default_threads =
      (bw_intra_node * num_channels <= kPciBandwidth) ? 256 : kLL128NumThreads;

  int warp_size = gpu_device_info.threads_per_warp();
  int num_threads = GetNumThreads(warp_size, kLL128NumThreads / 4,
                                  kLL128NumThreads, default_threads);

  // Since channels are pipelined together, compute time will only occur as in a
  // single channel.
  absl::Duration compute_time_per_channel =
      ComputeTime(gpu_device_info,
                  cost_analysis->flop_count(instr) / num_channels, num_threads);
  total_time += compute_time_per_channel;

  uint32_t supported_p2p = CheckIfNvlinkSupportsP2P();

  if (supported_p2p == 0) {
    VLOG(8) << "Nvlink doesn't support p2p communication. Model will "
               "continue using default system bandwidth.";
  } else {
    VLOG(8) << "Nvlink supports p2p communication, setting intra node "
               "bandwidth to nvlink bw.";
    bw_intra_node = GetNvlinkBw(compute_cap);
  }

  double bus_bandwidth = bw_intra_node * num_channels;

  // Get per channel LL128 ring bandwidth
  double per_channel_ring_ll128_Bw =
      GetMaxSysBwFromGpu(compute_cap, kPerChannelMaxRingLL128Bandwidths.data());

  bus_bandwidth = std::min(bus_bandwidth * kRingAlgorithmDiscountFactor,
                           num_channels * per_channel_ring_ll128_Bw);
  double actual_bandwidth = bus_bandwidth * cost_analysis->ScalingRatio(instr);

  absl::Duration communication_time = absl::Microseconds(
      cost_analysis->bytes_accessed(instr) / (1e6 * actual_bandwidth));
  total_time += communication_time;
  return total_time;
}

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeCollectiveTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info) {
  if (cost_analysis->NumOfDevices(instr) == 1) {
    VLOG(8) << "Returning only kernel launch overhead for a single partition.";
    return kNcclKernelLaunchOverhead;
  }

  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::ZeroDuration();
  }
  switch (instr.opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
      return ComputeAllreduceTime(instr, cost_analysis, gpu_device_info);
    default: {
      LOG(WARNING)
          << "Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      return kNcclKernelLaunchOverhead;
    }
  }
}

}  // namespace gpu
}  // namespace xla

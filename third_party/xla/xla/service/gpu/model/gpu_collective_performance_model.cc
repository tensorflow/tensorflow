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

#include "xla/service/gpu/model/gpu_collective_performance_model.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/time/time.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
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
    case se::CudaComputeCapability::kVolta:
      return bandwidths_table[0];
    case se::CudaComputeCapability::kAmpere:
      return bandwidths_table[1];
    case se::CudaComputeCapability::kHopper:
      return bandwidths_table[2];
    case se::CudaComputeCapability::kBlackwell:
      return bandwidths_table[3];
    default:
      return bandwidths_table[4];
  }
}

}  // namespace

// Returns NVLink bw in GB/s
/*static*/
float GpuPerformanceWithCollectiveModel::GetNvlinkBw(
    se::CudaComputeCapability compute_capability) {
  return compute_capability.IsAtLeast(se::CudaComputeCapability::kHopper)
             ? kSm90NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::kAmpere)
             ? kSm80NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::kVolta)
             ? kSm70NvlinkBandwidth
         : compute_capability.IsAtLeast(se::CudaComputeCapability::kPascal)
             ? kSm60NvlinkBandwidth
             : kSm80NvlinkBandwidth;
}

/*static*/ bool GpuPerformanceWithCollectiveModel::InitNvml() {
#if GOOGLE_CUDA && defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE)
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
      {(void**)&xla_nvmlSystemGetNVMLVersion, "nvmlSystemGetNVMLVersion"},
  };
#if GOOGLE_CUDA && CUDA_VERSION >= 12040
  symbols.push_back({(void**)&xla_nvmlDeviceGetHandleByPciBusId_v2,
                     "nvmlDeviceGetHandleByPciBusId_v2"});
  symbols.push_back({(void**)&xla_nvmlDeviceGetGpuFabricInfoV,
                     "nvmlDeviceGetGpuFabricInfoV"});
#endif  // CUDA_VERSION >= 12040
  for (SymbolEntry se : symbols) {
    *se.functor = dlsym(libhandle, se.name);
    if (*se.functor == nullptr) {
      const char* dlsym_error = dlerror();
      if (dlsym_error) {
        VLOG(0) << "Error: " << dlsym_error;
      }
    }
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
  CHECK(nvlink_cap_result == NVML_SUCCESS ||
        nvlink_cap_result == NVML_ERROR_NOT_SUPPORTED);
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

  int num_speeds = compute_cap.major >= se::CudaComputeCapability::kHopper
                       ? size_of_sm90_speed_array
                       : size_of_speed_array;
  const double* speeds = compute_cap.major >= se::CudaComputeCapability::kHopper
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
  absl::Duration compute_time_per_channel = ComputeTime(
      gpu_device_info, cost_analysis->flop_count(instr) / num_channels,
      /*num_blocks=*/num_channels, /*num_threads_per_block=*/num_threads);
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

  absl::Duration communication_time = absl::Milliseconds(
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

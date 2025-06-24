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
#include <array>
#include <cstdint>
#include <cstdlib>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/time/time.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/nvml/include/nvml.h"
#endif  // GOOGLE_CUDA
namespace xla {
namespace gpu {

namespace {

// Different algorithms that can be used to perform the collective.
enum class CollectiveAlgo {
  RING = 0,
  TREE,
};

struct CudaBandwidthSettings {
  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  static constexpr std::array<double, 5> kLowLatencyMaxBandwidths = {
      39.0 /* Volta */,      87.7 /* Ampere */,    141.0 /* Hopper */,
      141.0 /* Blackwell */, 141.0 /* next-gen */,
  };

  const std::vector<double>& GetIntraNodeBandwidths() const {
    // Different tiers for intra-node bandwidth.
    static const std::vector<double>* kIntraNodeSpeeds =
        new std::vector<double>{3.0,  4.0,  5.0,  6.0,  7.0,  9.0, 10.0,
                                12.0, 15.0, 18.0, 20.0, 30.0, 40.0};
    // SM90 has different bandwidths.
    static std::vector<double>* kIntraNodeSpeedsSm90 = new std::vector<double>{
        3.0, 6.0, 12.0, 15.0, 20.0, 24.0, 30.0, 40.0, 60.0};
    return compute_capability.major >= se::CudaComputeCapability::kHopper
               ? *kIntraNodeSpeedsSm90
               : *kIntraNodeSpeeds;
  }

  float GetMaxSysBwFromGpu(const double* bandwidths_table) const {
    switch (compute_capability.major) {
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

  // Returns NVLink bw in GB/s
  float GetNvlinkBw() const {
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

  // Max bandwidth in GB/s for ring low latency 128 algorithm per channel on a
  // single-node
  static constexpr std::array<double, 5> kPerChannelMaxRingLL128Bandwidths = {
      20.0 /* Volta */,     20.0 /* Ampere */,   36.7 /* Hopper */,
      36.7 /* Blackwell */, 36.7 /* next-gen */,
  };

  // Nvlink unidirectional bandwidth for different compute cap. Note this is per
  // lane bandwidth.
  static constexpr double kSm60NvlinkBandwidth = 18.0;
  static constexpr double kSm70NvlinkBandwidth = 20.0;
  static constexpr double kSm80NvlinkBandwidth = 20.0;
  static constexpr double kSm90NvlinkBandwidth = 20.0;

  // PCIE bandwidth for PCI Gen3 x16
  static constexpr double kPciBandwidth = 12.0;

  // Discount factor for ring algorithm
  static constexpr double kRingAlgorithmDiscountFactor = 0.92;

  // Maximum number of channels allowed by NCCL
  static constexpr int64_t kMaxNumChannelsRing = 16;

  // ll128 is by default enabled for Volta, Ampere and Hopper, ll128 by default
  // launches 640 threads.
  static constexpr int64_t kLL128NumThreads = 640;

  stream_executor::CudaComputeCapability compute_capability;
};

struct RocmBandwidthSettings {
  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  static constexpr std::array<double, 4> kLowLatencyMaxBandwidths = {
      300.0 /* MI100 (8x Infinity Fabric @ ~37.5GB/s each) */,
      600.0 /* MI200 (8x IF @ ~75GB/s each) */,
      896.0 /* MI300 (8x IF @ ~112GB/s each) */,
      896.0 /* next_gen (same as MI300 for now) */,
  };

  const std::vector<double>& GetIntraNodeBandwidths() const {
    // Different tiers for intra-node bandwidth based on Infinity Fabric
    // capabilities Values in GB/s

    // MI300 series (Instinct MI300) - up to 896GB/s (8x112GB/s)
    static const std::vector<double>* intraNodeSpeedsMi300 =
        new std::vector<double>{32.0,  56.0,  112.0, 224.0, 336.0,
                                448.0, 560.0, 672.0, 784.0, 896.0};

    // MI200 series (Instinct MI200/MI250) - up to 600GB/s (8x75GB/s)
    static const std::vector<double>* intraNodeSpeedsMi200 =
        new std::vector<double>{32.0,  75.0,  150.0, 225.0, 300.0,
                                375.0, 450.0, 525.0, 600.0};

    // MI100 (Instinct MI100) - up to 300GB/s (8x37.5GB/s)
    static const std::vector<double>* intraNodeSpeedsMi100 =
        new std::vector<double>{32.0,  37.5,  75.0,  112.5, 150.0,
                                187.5, 225.0, 262.5, 300.0};

    if (compute_capability.gfx9_mi300_series()) {
      return *intraNodeSpeedsMi300;
    }
    if (compute_capability.gfx9_mi200()) {
      return *intraNodeSpeedsMi200;
    }
    if (compute_capability.gfx9_mi100()) {
      return *intraNodeSpeedsMi100;
    }

    // Default to MI300 speeds for unknown architectures
    return *intraNodeSpeedsMi300;
  }

  // Max bandwidth in GB/s for ring low latency 128 algorithm per channel on a
  // single-node
  static constexpr std::array<double, 5> kPerChannelMaxRingLL128Bandwidths = {
      37.5 /* MI100 (per IF link) */,
      75.0 /* MI200 (per IF link) */,
      112.0 /* MI300 (per IF link) */,
      112.0 /* next_gen */,
  };

  float GetMaxSysBwFromGpu(const double* bandwidths_table) const {
    if (compute_capability.gfx9_mi100()) {
      return bandwidths_table[0];
    }
    if (compute_capability.gfx9_mi200()) {
      return bandwidths_table[1];
    }
    if (compute_capability.gfx9_mi300_series()) {
      return bandwidths_table[2];
    }
    return bandwidths_table[3];
  }

  float GetNvlinkBw() const {
    if (compute_capability.gfx9_mi100()) {
      return kMi100InfinityFabricBandwidth;
    }
    if (compute_capability.gfx9_mi200()) {
      return kMi200InfinityFabricBandwidth;
    }
    if (compute_capability.gfx9_mi300_series()) {
      return kMi300InfinityFabricBandwidth;
    }
    return kMi300InfinityFabricBandwidth;
  }

  // Infinity Fabric unidirectional bandwidth per link in GB/s
  static constexpr double kMi100InfinityFabricBandwidth = 37.5;
  static constexpr double kMi200InfinityFabricBandwidth = 75.0;
  static constexpr double kMi300InfinityFabricBandwidth = 112.0;

  // PCIe bandwidth for PCI Gen4 x16 (approximate)
  static constexpr double kPciBandwidth = 32.0;

  // Discount factor for ring algorithm (based on ROCm NCCL implementation)
  static constexpr double kRingAlgorithmDiscountFactor = 0.90;

  // Default number of threads for ROCm NCCL
  static constexpr int64_t kLL128NumThreads = 512;

  stream_executor::RocmComputeCapability compute_capability;
};

template <typename BandwidthSettings>
float GetMaxPerChannelRingLL128Bandwidth(
    const BandwidthSettings& bandwidth_settings) {
  return bandwidth_settings.GetMaxSysBwFromGpu(
      bandwidth_settings.kPerChannelMaxRingLL128Bandwidths.data());
}

template <typename BandwidthSettings>
float GetMaxLowLatencyBandwidth(const BandwidthSettings& bandwidth_settings) {
  auto speeds = bandwidth_settings.GetIntraNodeBandwidths();
  auto max_sys_bw = bandwidth_settings.GetMaxSysBwFromGpu(
      bandwidth_settings.kLowLatencyMaxBandwidths.data());
  auto it = std::lower_bound(std::begin(speeds), std::end(speeds), max_sys_bw);
  CHECK(it != std::cend(speeds));
  return *it;
}

static constexpr absl::Duration kNcclKernelLaunchOverhead =
    absl::Microseconds(5);

int64_t GetNcclMaxNumChannels(CollectiveAlgo algorithm) {
  int64_t max_nchannels = 0;
  switch (algorithm) {
      // Tree and Ring algos share the same max channel number.
    case CollectiveAlgo::RING:
    case CollectiveAlgo::TREE:
      max_nchannels = CudaBandwidthSettings::kMaxNumChannelsRing;
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

int64_t GetMinNumberOfChannels(CollectiveAlgo algorithm) {
  int64_t min_nchannels = 0;
  switch (algorithm) {
      // Tree and Ring algos share the same min channel number.
    case CollectiveAlgo::RING:
    case CollectiveAlgo::TREE:
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

template <typename GpuBandwidthSettings>
absl::Duration ComputeAllreduceTimeImpl(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info,
    const GpuBandwidthSettings& bandwidth_settings) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  absl::Duration total_time = kNcclKernelLaunchOverhead;

  float bw_intra_node = GetMaxLowLatencyBandwidth(bandwidth_settings);
  int64_t num_devices = cost_analysis->NumOfDevices(instr);

  int64_t min_nchannels =
      std::max(num_devices, GetMinNumberOfChannels(CollectiveAlgo::RING));
  int64_t num_channels =
      std::max(min_nchannels, GetNcclMaxNumChannels(CollectiveAlgo::RING));
  int default_threads =
      (bw_intra_node * num_channels <= bandwidth_settings.kPciBandwidth)
          ? 256
          : bandwidth_settings.kLL128NumThreads;

  int warp_size = gpu_device_info.threads_per_warp();
  int num_threads =
      GetNumThreads(warp_size, bandwidth_settings.kLL128NumThreads / 4,
                    bandwidth_settings.kLL128NumThreads, default_threads);

  // Since channels are pipelined together, compute time will only occur as in a
  // single channel.
  absl::Duration compute_time_per_channel =
      GpuPerformanceWithCollectiveModel::ComputeTime(
          gpu_device_info, cost_analysis->flop_count(instr) / num_channels,
          /*num_blocks=*/num_channels, /*num_threads_per_block=*/num_threads);
  total_time += compute_time_per_channel;

  uint32_t supported_p2p =
      GpuPerformanceWithCollectiveModel::CheckIfNvlinkSupportsP2P();

  if (supported_p2p == 0) {
    VLOG(8) << "Nvlink doesn't support p2p communication. Model will "
               "continue using default system bandwidth.";
  } else {
    VLOG(8) << "Nvlink supports p2p communication, setting intra node "
               "bandwidth to nvlink bw.";
    bw_intra_node = bandwidth_settings.GetNvlinkBw();
  }

  double bus_bandwidth = bw_intra_node * num_channels;

  // Get per channel LL128 ring bandwidth
  double per_channel_ring_ll128_Bw =
      GetMaxPerChannelRingLL128Bandwidth(bandwidth_settings);

  bus_bandwidth =
      std::min(bus_bandwidth * bandwidth_settings.kRingAlgorithmDiscountFactor,
               num_channels * per_channel_ring_ll128_Bw);
  double actual_bandwidth = bus_bandwidth * cost_analysis->ScalingRatio(instr);

  absl::Duration communication_time = absl::Milliseconds(
      cost_analysis->bytes_accessed(instr) / (1e6 * actual_bandwidth));
  total_time += communication_time;
  return total_time;
}

CudaBandwidthSettings CreateSettings(
    const stream_executor::CudaComputeCapability& cc) {
  return CudaBandwidthSettings{cc};
}

RocmBandwidthSettings CreateSettings(
    const stream_executor::RocmComputeCapability& cc) {
  return RocmBandwidthSettings{cc};
}

}  // namespace

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
#elif TENSORFLOW_USE_ROCM
  return true;
#else
  return false;
#endif  // GOOGLE_CUDA
}

/*static*/ bool GpuPerformanceWithCollectiveModel::ShutdownNvml() {
#if GOOGLE_CUDA
  nvmlReturn_t shutdown_result = xla_nvmlShutdown();
  return shutdown_result == NVML_SUCCESS;
#elif TENSORFLOW_USE_ROCM
  return true;
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
  const auto visitor = [&](const auto& cc) {
    return ComputeAllreduceTimeImpl(instr, cost_analysis, gpu_device_info,
                                    CreateSettings(cc));
  };
  return std::visit(visitor, gpu_device_info.gpu_compute_capability());
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

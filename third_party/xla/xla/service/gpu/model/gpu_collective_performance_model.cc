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
#include <vector>

#include "absl/base/no_destructor.h"
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

namespace xla {
namespace gpu {
namespace {

struct CudaBandwidthSettings {
  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  static constexpr std::array<double, 5> kLowLatencyMaxBandwidths = {
      39.0 /* Volta */,      87.7 /* Ampere */,    141.0 /* Hopper */,
      141.0 /* Blackwell */, 141.0 /* next-gen */,
  };

  const std::vector<double>& GetIntraNodeBandwidths() const {
    // Different tiers for intra-node bandwidth.
    static const absl::NoDestructor<std::vector<double>> kIntraNodeSpeeds(
        {3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 30.0,
         40.0});
    // SM90 has different bandwidths.
    static const absl::NoDestructor<std::vector<double>> kIntraNodeSpeedsSm90(
        {3.0, 6.0, 12.0, 15.0, 20.0, 24.0, 30.0, 40.0, 60.0});
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
    switch (compute_capability.major) {
      case se::CudaComputeCapability::kBlackwell:
        return kSm100NvlinkBandwidth;
      case se::CudaComputeCapability::kHopper:
        return kSm90NvlinkBandwidth;
      case se::CudaComputeCapability::kAmpere:
        return kSm80NvlinkBandwidth;
      case se::CudaComputeCapability::kVolta:
        return kSm70NvlinkBandwidth;
      case se::CudaComputeCapability::kPascal:
        return kSm60NvlinkBandwidth;
      default:
        LOG(WARNING) << "NVLink bandwidth for " << compute_capability.ToString()
                     << "unknown. Assumes Blackwell.";
        return kSm100NvlinkBandwidth;
    }
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
  static constexpr double kSm100NvlinkBandwidth = 40.0;

  // Discount factor for ring algorithm
  static constexpr double kRingAlgorithmDiscountFactor = 0.92;

  // Maximum number of channels allowed by NCCL
  static constexpr int64_t kMaxNumChannelsRing = 32;

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
    static const absl::NoDestructor<std::vector<double>> intraNodeSpeedsMi300(
        {32.0, 56.0, 112.0, 224.0, 336.0, 448.0, 560.0, 672.0, 784.0, 896.0});

    // MI200 series (Instinct MI200/MI250) - up to 600GB/s (8x75GB/s)
    static const absl::NoDestructor<std::vector<double>> intraNodeSpeedsMi200(
        {32.0, 75.0, 150.0, 225.0, 300.0, 375.0, 450.0, 525.0, 600.0});

    // MI100 (Instinct MI100) - up to 300GB/s (8x37.5GB/s)
    static const absl::NoDestructor<std::vector<double>> intraNodeSpeedsMi100(
        {32.0, 37.5, 75.0, 112.5, 150.0, 187.5, 225.0, 262.5, 300.0});

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
  if (it == speeds.end()) {
    return speeds.back();
  }
  return *it;
}

static constexpr absl::Duration kNcclKernelLaunchOverhead =
    absl::Microseconds(5);

int64_t GetNcclMaxNumChannels() {
  int64_t max_nchannels = CudaBandwidthSettings::kMaxNumChannelsRing;

  const char* env = std::getenv("NCCL_MAX_NCHANNELS");
  if (env != nullptr) {
    int64_t max_nchannels_from_env;
    if (absl::SimpleAtoi(env, &max_nchannels_from_env)) {
      max_nchannels = std::min(max_nchannels_from_env, max_nchannels);
    }
  }
  return max_nchannels;
}

int64_t GetMinNumberOfChannels() {
  int64_t min_nchannels = 1;
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

  int64_t min_nchannels = std::max(num_devices, GetMinNumberOfChannels());
  int64_t num_channels = std::max(min_nchannels, GetNcclMaxNumChannels());
  int64_t pcie_bandwidth_gbps =
      gpu_device_info.pcie_bandwidth() / 1024 / 1024 / 1024;
  int default_threads = (bw_intra_node * num_channels <= pcie_bandwidth_gbps)
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

  if (gpu_device_info.device_interconnect_info().active_links) {
    VLOG(8) << "Nvlink supports p2p communication, setting intra node "
               "bandwidth to nvlink bw.";
    bw_intra_node = bandwidth_settings.GetNvlinkBw();
    num_channels =
        std::min(static_cast<int64_t>(
                     gpu_device_info.device_interconnect_info().active_links),
                 num_channels);
  } else {
    VLOG(8) << "Nvlink doesn't support p2p communication. Model will "
               "continue using default system bandwidth.";
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

/*static*/ absl::Duration
GpuPerformanceWithCollectiveModel::ComputeAllreduceTime(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  if (auto ptr =
          gpu_device_info.gpu_compute_capability().cuda_compute_capability()) {
    return ComputeAllreduceTimeImpl(instr, cost_analysis, gpu_device_info,
                                    CreateSettings(*ptr));
  }
  return ComputeAllreduceTimeImpl(
      instr, cost_analysis, gpu_device_info,
      CreateSettings(
          *gpu_device_info.gpu_compute_capability().rocm_compute_capability()));
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

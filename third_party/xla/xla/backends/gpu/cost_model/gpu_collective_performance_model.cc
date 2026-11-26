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

#include "xla/backends/gpu/cost_model/gpu_collective_performance_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/cost_model/gpu_hlo_cost_analysis.h"
#include "xla/backends/gpu/cost_model/ici_bandwidth.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

struct CudaBandwidthSettings {
  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  // Rubin and fallback entries are per CTA.
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/tuning/cost_model.cc#L192-L199
  static constexpr std::array<double, 6> kLowLatencyMaxBandwidths = {
      39.0 /* Volta */,          87.7 /* Ampere */, 141.0 /* Hopper */,
      2 * 141.2 /* Blackwell */, 2.0 /* Rubin */,   2.0 /* fallback */,
  };

  const std::vector<double>& GetIntraNodeBandwidths() const {
    // Different tiers for intra-node bandwidth.
    // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/search.cc#L1193
    static const absl::NoDestructor<std::vector<double>> kIntraNodeSpeeds(
        {3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0, 30.0,
         40.0});
    // SM90 has different bandwidths.
    // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/search.cc#L1199
    static const absl::NoDestructor<std::vector<double>> kIntraNodeSpeedsSm90(
        {3.0, 6.0, 11.0, 12.0, 15.0, 20.0, 24.0, 30.0, 40.0, 50.0, 60.0});
    // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/search.cc#L1205
    static const absl::NoDestructor<std::vector<double>> kIntraNodeSpeedsSm100(
        {18.0, 19.0, 20.0, 24.0, 30.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0,
         90.0});
    // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/search.cc#L1212
    static const absl::NoDestructor<std::vector<double>> kIntraNodeSpeedsSm107(
        {17.55, 23.4, 35.1, 46.8, 70.2, 93.6, 140.4, 187.2, 280.8});
    switch (compute_capability.major) {
      case se::CudaComputeCapability::kPascal:
      case se::CudaComputeCapability::kVolta:
      case se::CudaComputeCapability::kAmpere:
        return *kIntraNodeSpeeds;
      case se::CudaComputeCapability::kHopper:
        return *kIntraNodeSpeedsSm90;
      case se::CudaComputeCapability::kBlackwell:
      case se::CudaComputeCapability::kBlackwell_11:
      case se::CudaComputeCapability::kBlackwell_12:
        return compute_capability.IsRubin() ? *kIntraNodeSpeedsSm107
                                            : *kIntraNodeSpeedsSm100;
      default:
        return *kIntraNodeSpeedsSm107;
    }
  }

  float GetMaxSysBwFromGpu(const double* bandwidths_table) const {
    switch (compute_capability.major) {
      // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/tuning/tuning_general.cc#L27-L39
      case se::CudaComputeCapability::kPascal:
      case se::CudaComputeCapability::kVolta:
        return bandwidths_table[0];
      case se::CudaComputeCapability::kAmpere:
        return bandwidths_table[1];
      case se::CudaComputeCapability::kHopper:
        return bandwidths_table[2];
      case se::CudaComputeCapability::kBlackwell:
      case se::CudaComputeCapability::kBlackwell_11:
      case se::CudaComputeCapability::kBlackwell_12:
        return bandwidths_table[compute_capability.IsRubin() ? 4 : 3];
      default:
        return bandwidths_table[5];
    }
  }

  float GetLowLatencyMaxBandwidth(int64_t graph_channels,
                                  int64_t bandwidth_scale) const {
    float max_sys_bw = GetMaxSysBwFromGpu(kLowLatencyMaxBandwidths.data());
    if (UsesPerCtaCostModel()) {
      // NCCL aggregates the per-CTA cap over graph channels and CTAs per
      // channel.
      // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/tuning/pipeline_model/ring.cc#L20-L22
      max_sys_bw *= graph_channels * bandwidth_scale;
    }
    return max_sys_bw;
  }

  // Returns NVLink bw in GB/s
  float GetNvlinkBw() const {
    return GetIciBandwidthPerLaneGbps(compute_capability);
  }

  // Max bandwidth in GB/s for ring low latency 128 algorithm per channel on a
  // single-node
  // Rubin and fallback entries are per CTA.
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/tuning/cost_model.cc#L201-L208
  static constexpr std::array<double, 6> kPerChannelMaxRingLL128Bandwidths = {
      20.0 /* Volta */,     20.0 /* Ampere */, 36.7 /* Hopper */,
      40.0 /* Blackwell */, 21.6 /* Rubin */,  21.6 /* fallback */,
  };

  // Discount factor for ring algorithm
  // https://github.com/NVIDIA/nccl/blob/ea38312273e5b9a19a224c9ff4c10b7fcf441eaf/src/graph/tuning.cc#L179
  static constexpr double kRingAlgorithmDiscountFactor = 0.92;

  // NCCL's compute-channel ceiling.
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/include/device.h#L80
  static constexpr int64_t kMaxComputeChannels = 64;
  // Maximum number of Ring graph channels allowed by NCCL
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/init.cc#L1471
  static constexpr int64_t kMaxNumChannelsRing = kMaxComputeChannels / 2;

  // ll128 is by default enabled for Volta, Ampere and Hopper, ll128 by default
  // launches 640 threads.
  static constexpr int64_t kLL128NumThreads = 640;

  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/connect.cc#L500-L501
  bool UsesPerCtaCostModel() const {
    switch (compute_capability.major) {
      case se::CudaComputeCapability::kPascal:
      case se::CudaComputeCapability::kVolta:
      case se::CudaComputeCapability::kAmpere:
      case se::CudaComputeCapability::kHopper:
      case se::CudaComputeCapability::kBlackwell_11:
      case se::CudaComputeCapability::kBlackwell_12:
        return false;
      case se::CudaComputeCapability::kBlackwell:
        return compute_capability.IsRubin();
      default:
        return true;
    }
  }

  double GetRingBusBandwidth(int64_t active_links, double transport_ceiling,
                             double ll128_ceiling) const {
    return active_links && transport_ceiling < ll128_ceiling ? transport_ceiling
                                                             : ll128_ceiling;
  }

  int64_t GetRingBandwidthScale(int64_t graph_channels) const;

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

  float GetLowLatencyMaxBandwidth(int64_t /*graph_channels*/,
                                  int64_t /*bandwidth_scale*/) const {
    return GetMaxSysBwFromGpu(kLowLatencyMaxBandwidths.data());
  }

  float GetNvlinkBw() const {
    return GetIciBandwidthPerLaneGbps(compute_capability);
  }

  // Discount factor for ring algorithm (based on ROCm NCCL implementation)
  static constexpr double kRingAlgorithmDiscountFactor = 0.90;

  // Default number of threads for ROCm NCCL
  static constexpr int64_t kLL128NumThreads = 512;

  double GetRingBusBandwidth(int64_t /*active_links*/, double transport_ceiling,
                             double ll128_ceiling) const {
    return std::min(transport_ceiling, ll128_ceiling);
  }

  int64_t GetRingBandwidthScale(int64_t /*graph_channels*/) const { return 1; }

  stream_executor::RocmComputeCapability compute_capability;
};

int64_t CudaBandwidthSettings::GetRingBandwidthScale(
    int64_t graph_channels) const {
  if (!UsesPerCtaCostModel()) {
    return 1;
  }
  // Per-link NVLink bandwidth approximates NCCL's per-graph-channel bandwidth.
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/connect.cc#L502
  const double ring_graph_capacity = graph_channels * GetNvlinkBw();
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/connect.cc#L393
  constexpr int64_t kBandwidthTargetPerCta = 24;
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/connect.cc#L479
  int64_t compute_channels = std::min(graph_channels * 2, kMaxComputeChannels);
  const int64_t min_compute_channels = static_cast<int64_t>(
      std::ceil(ring_graph_capacity / kBandwidthTargetPerCta));
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/connect.cc#L504-L507
  while (compute_channels < min_compute_channels &&
         compute_channels < kMaxComputeChannels) {
    compute_channels = std::min(compute_channels * 2, kMaxComputeChannels);
  }
  // https://github.com/NVIDIA/nccl/blob/72a2f960a04e6a3e3ef1849a2563a2383f7baf46/src/graph/connect.cc#L511
  return CeilOfRatio(compute_channels, graph_channels);
}

template <typename BandwidthSettings>
float GetMaxPerChannelRingLL128Bandwidth(
    const BandwidthSettings& bandwidth_settings, int64_t bandwidth_scale) {
  float bandwidth = bandwidth_settings.GetMaxSysBwFromGpu(
      bandwidth_settings.kPerChannelMaxRingLL128Bandwidths.data());
  return bandwidth * bandwidth_scale;
}

template <typename BandwidthSettings>
float GetMaxLowLatencyBandwidth(const BandwidthSettings& bandwidth_settings,
                                int64_t graph_channels,
                                int64_t bandwidth_scale) {
  auto speeds = bandwidth_settings.GetIntraNodeBandwidths();
  auto max_sys_bw = bandwidth_settings.GetLowLatencyMaxBandwidth(
      graph_channels, bandwidth_scale);
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
absl::Duration ComputeCollectiveTimeImpl(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info,
    const GpuBandwidthSettings& bandwidth_settings) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  absl::Duration total_time = kNcclKernelLaunchOverhead;

  int64_t num_devices = cost_analysis->NumOfDevices(instr);

  int64_t min_nchannels = std::max(num_devices, GetMinNumberOfChannels());
  int64_t compute_channels = std::max(min_nchannels, GetNcclMaxNumChannels());
  const int64_t active_links =
      gpu_device_info.device_interconnect_info().active_links;
  const int64_t graph_channels = active_links
                                     ? std::min(compute_channels, active_links)
                                     : compute_channels;
  const int64_t bandwidth_scale =
      bandwidth_settings.GetRingBandwidthScale(graph_channels);
  float bw_intra_node = GetMaxLowLatencyBandwidth(
      bandwidth_settings, graph_channels, bandwidth_scale);
  int64_t pcie_bandwidth_gbps =
      gpu_device_info.pcie_bandwidth() / 1024 / 1024 / 1024;
  int default_threads =
      (bw_intra_node * compute_channels <= pcie_bandwidth_gbps)
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
          gpu_device_info, cost_analysis->flop_count(instr) / compute_channels,
          /*num_blocks=*/compute_channels,
          /*num_threads_per_block=*/num_threads);
  total_time += compute_time_per_channel;

  if (active_links) {
    VLOG(8) << "Nvlink supports p2p communication, setting intra node "
               "bandwidth to nvlink bw.";
    bw_intra_node = bandwidth_settings.GetNvlinkBw();
  } else {
    VLOG(8) << "Nvlink doesn't support p2p communication. Model will "
               "continue using default system bandwidth.";
  }

  double bus_bandwidth = bw_intra_node * graph_channels;

  // Get per channel LL128 ring bandwidth
  double per_channel_ring_ll128_Bw =
      GetMaxPerChannelRingLL128Bandwidth(bandwidth_settings, bandwidth_scale);

  bus_bandwidth = bandwidth_settings.GetRingBusBandwidth(
      active_links,
      bus_bandwidth * bandwidth_settings.kRingAlgorithmDiscountFactor,
      graph_channels * per_channel_ring_ll128_Bw);
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
GpuPerformanceWithCollectiveModel::ComputeCollectiveTimeForRing(
    const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info) {
  // We use nccl group call to launch multiple allreduces so launch overhead
  // only occurs once.
  if (auto ptr =
          gpu_device_info.gpu_compute_capability().cuda_compute_capability()) {
    return ComputeCollectiveTimeImpl(instr, cost_analysis, gpu_device_info,
                                     CreateSettings(*ptr));
  }
  return ComputeCollectiveTimeImpl(
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
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kReduceScatter:
      return ComputeCollectiveTimeForRing(instr, cost_analysis,
                                          gpu_device_info);
    case HloOpcode::kAsyncStart: {
      auto* async_start = Cast<HloAsyncStartInstruction>(&instr);
      switch (async_start->async_wrapped_opcode()) {
        case HloOpcode::kReduceScatter:
          return ComputeCollectiveTimeForRing(instr, cost_analysis,
                                              gpu_device_info);
        default:
          break;
      }
      LOG(WARNING)
          << "Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      return kNcclKernelLaunchOverhead;
    }
    default: {
      LOG(WARNING)
          << "Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      return kNcclKernelLaunchOverhead;
    }
  }
}

/*static*/ absl::StatusOr<double>
GpuPerformanceWithCollectiveModel::GetIciBandwidthPerLaneGbps(
    const se::DeviceDescription& gpu_device_info) {
  return xla::gpu::GetIciBandwidthPerLaneGbps(gpu_device_info);
}

}  // namespace gpu
}  // namespace xla

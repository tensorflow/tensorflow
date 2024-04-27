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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_

#include <array>
#include <cstdint>

#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

#if GOOGLE_CUDA
#include <dlfcn.h>

#include "third_party/gpus/cuda/nvml/include/nvml.h"
// Below is a list of function pointers to be used
// for querying device properties through nvml library.
#define NVML_FUNCTOR(name, rettype, args) \
  inline rettype(*xla_##name) args = nullptr;

NVML_FUNCTOR(nvmlInit, nvmlReturn_t, ())
NVML_FUNCTOR(nvmlShutdown, nvmlReturn_t, ())
NVML_FUNCTOR(nvmlDeviceGetHandleByIndex, nvmlReturn_t,
             (unsigned int index, nvmlDevice_t* device))
NVML_FUNCTOR(nvmlDeviceGetNvLinkCapability, nvmlReturn_t,
             (nvmlDevice_t device, unsigned int link,
              nvmlNvLinkCapability_t capability, unsigned int* capResult))

#endif

namespace xla {
namespace gpu {

class GpuPerformanceWithCollectiveModel : public GpuPerformanceModelBase {
 public:
  // Different algorithms that can be used to perform the collective.
  enum CollectiveAlgo {
    RING = 0,
    TREE,
  };

  // Table for max system bandwidths GB/s for using NCCL's low latency
  // algorithm. This is used for intra-node estimate.
  static constexpr std::array<double, 3> kLowLatencyMaxBandwidths = {
      39.0 /* Volta*/, 87.7 /* Ampere*/, 87.7 /* Hopper*/
  };

  // Max bandwidth in GB/s for ring low latency 128 algorithm per channel on a
  // single-node
  static constexpr std::array<double, 3> kPerChannelMaxRingLL128Bandwidths = {
      20.0 /* Volta */,
      20.0 /* Ampere */,
      36.7 /* Hopper */,
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

  // Different tiers for intra-node bandwidth.
  static constexpr std::array<double, 13> kIntraNodeSpeeds = {
      40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0};
  // SM90 has different bandwidths.
  static constexpr std::array<double, 9> kIntraNodeSpeedsSm90 = {
      60.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 6.0, 3.0};

  // Maximum number of channels allowed by NCCL
  static constexpr int64_t kMaxNumChannelsRing = 16;

  // ll128 is by default enabled for Volta, Ampere and Hopper, ll128 by default
  // launches 640 threads.
  static constexpr int64_t kLL128NumThreads = 640;

  static constexpr absl::Duration kNcclKernelLaunchOverhead =
      absl::Microseconds(5);

  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const se::DeviceDescription& gpu_device_info);

  // Returns NVLink bw in GB/s
  static float GetNvlinkBw(se::CudaComputeCapability compute_capability);

  // Initialize nvml library.
  static bool InitNvml();

  // Shut down nvml library.
  static bool ShutdownNvml();

  // This checks if the nvlink supports direct P2P communication,
  // If not, we will use PCIE bandwidth to estimate latency.
  static uint32_t CheckIfNvlinkSupportsP2P();

 private:
  static absl::Duration ComputeAllreduceTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const se::DeviceDescription& gpu_device_info);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_

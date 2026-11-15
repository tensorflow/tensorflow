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

#ifndef XLA_SERVICE_GPU_MODEL_ICI_BANDWIDTH_H_
#define XLA_SERVICE_GPU_MODEL_ICI_BANDWIDTH_H_

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {

// Unidirectional per-lane NVLink bandwidth in GB/s, per CUDA compute
// capability. Shared source of truth for the NCCL/XLA performance model and
// for runtime cost calculations.
inline constexpr double kSm60NvlinkBandwidthGbps = 18.0;
inline constexpr double kSm70NvlinkBandwidthGbps = 20.0;
inline constexpr double kSm80NvlinkBandwidthGbps = 20.0;
inline constexpr double kSm90NvlinkBandwidthGbps = 20.0;
inline constexpr double kSm100NvlinkBandwidthGbps = 40.0;

// Unidirectional per-lane Infinity Fabric bandwidth in GB/s, per ROCm
// Instinct accelerator family.
inline constexpr double kMi100InfinityFabricBandwidthGbps = 37.5;
inline constexpr double kMi200InfinityFabricBandwidthGbps = 75.0;
inline constexpr double kMi300InfinityFabricBandwidthGbps = 112.0;

// Returns the unidirectional inter-chip interconnect (ICI) bandwidth per
// lane (GB/s) for the given CUDA compute capability. Corresponds to the
// peer-to-peer NVLink lane bandwidth.
inline float GetIciBandwidthPerLaneGbps(
    const stream_executor::CudaComputeCapability& cc) {
  switch (cc.major) {
    case stream_executor::CudaComputeCapability::kBlackwell:
      return kSm100NvlinkBandwidthGbps;
    case stream_executor::CudaComputeCapability::kHopper:
      return kSm90NvlinkBandwidthGbps;
    case stream_executor::CudaComputeCapability::kAmpere:
      return kSm80NvlinkBandwidthGbps;
    case stream_executor::CudaComputeCapability::kVolta:
      return kSm70NvlinkBandwidthGbps;
    case stream_executor::CudaComputeCapability::kPascal:
      return kSm60NvlinkBandwidthGbps;
    default:
      LOG(WARNING) << "NVLink bandwidth for " << cc.ToString()
                   << " unknown. Assumes Blackwell.";
      return kSm100NvlinkBandwidthGbps;
  }
}

// Returns the unidirectional inter-chip interconnect (ICI) bandwidth per
// lane (GB/s) for the given ROCm compute capability. Corresponds to the
// peer-to-peer Infinity Fabric lane bandwidth.
inline float GetIciBandwidthPerLaneGbps(
    const stream_executor::RocmComputeCapability& cc) {
  if (cc.gfx9_mi300_series()) {
    return kMi300InfinityFabricBandwidthGbps;
  }
  if (cc.gfx9_mi200()) {
    return kMi200InfinityFabricBandwidthGbps;
  }
  if (cc.gfx9_mi100()) {
    return kMi100InfinityFabricBandwidthGbps;
  }
  return kMi300InfinityFabricBandwidthGbps;
}

// Returns the unidirectional inter-chip interconnect bandwidth per lane
// (GB/s) for `device_info`, or an error if the device is neither CUDA nor
// ROCm.
inline absl::StatusOr<double> GetIciBandwidthPerLaneGbps(
    const stream_executor::DeviceDescription& device_info) {
  if (const auto* cuda_cc =
          device_info.gpu_compute_capability().cuda_compute_capability()) {
    return GetIciBandwidthPerLaneGbps(*cuda_cc);
  }
  if (const auto* rocm_cc =
          device_info.gpu_compute_capability().rocm_compute_capability()) {
    return GetIciBandwidthPerLaneGbps(*rocm_cc);
  }
  return absl::InvalidArgumentError(
      "GetIciBandwidthPerLaneGbps: unsupported compute capability "
      "(neither CUDA nor ROCm)");
}

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_ICI_BANDWIDTH_H_

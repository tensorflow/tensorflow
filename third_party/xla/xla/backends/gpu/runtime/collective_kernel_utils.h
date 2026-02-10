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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_UTILS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
// Class with static methods used for collective kernels: kernels which are
// launched symmetrically on multiple devices and can access each other's data.
// Methods from this class used both by XLA and Mosaic, so be careful when
// changing the API.
class CollectiveKernelUtils {
 public:
  // Launches a cross-GPU barrier synchronization.
  static absl::Status LaunchMultiGpuBarrier(
      stream_executor::Stream* stream, int64_t num_devices, RankId rank,
      std::vector<stream_executor::DeviceAddressBase> barrier_addresses,
      stream_executor::DeviceAddressBase local_barrier_signal_value);
};
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_UTILS_H_

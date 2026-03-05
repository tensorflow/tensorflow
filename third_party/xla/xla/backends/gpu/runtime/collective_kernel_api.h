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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_API_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_API_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// This file contains collective kernels related method shared between XLA and
// Mosaic. Be careful when changing the API.
//
// Collective kernels are kernels which are launched simultaneously on multiple
// devices, and have an access to each other device's memory.

// Launches a cross-GPU barrier synchronization.
absl::Status LaunchMultiGpuBarrier(
    stream_executor::Stream* stream, int64_t num_devices, RankId rank,
    const std::vector<stream_executor::DeviceAddressBase>& barrier_addresses,
    stream_executor::DeviceAddressBase local_barrier_signal_value);

// Returns the size of the barrier signal buffer in bytes.
size_t GetMultiGpuBarrierSignalBufferSize();

// Returns the size of the barrier signal value in bytes.
size_t GetMultiGpuBarrierSignalValueSize();

// Collect the pointers to the parameters at the peer devices.
// The size of the returned vector is num_parameters * num_devices.
absl::StatusOr<std::vector<void*>> CollectParamToPeers(
    const GpuCliqueKey& clique_key, RankId rank,
    stream_executor::Stream* stream,
    std::vector<stream_executor::DeviceAddressBase> parameters);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_API_H_

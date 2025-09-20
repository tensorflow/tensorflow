/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_SDC_XOR_CHECKSUM_KERNEL_CUDA_H_
#define XLA_STREAM_EXECUTOR_CUDA_SDC_XOR_CHECKSUM_KERNEL_CUDA_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/sdc_log.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {

// Trait for a kernel that computes the checksum of given input buffer and
// appends it to the SDC log.
//
// This kernel MUST execute on a single thread block.
struct SdcXorChecksumKernel {
  using KernelType =
      TypedKernel<uint32_t, DeviceMemory<uint8_t>, uint64_t,
                  DeviceMemory<SdcLogHeader>, DeviceMemory<SdcLogEntry>>;
};

absl::StatusOr<KernelLoaderSpec> GetSdcXorChecksumKernelSpec();

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_SDC_XOR_CHECKSUM_KERNEL_CUDA_H_

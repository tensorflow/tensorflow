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

#include <cstdint>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

namespace xla::gpu {
namespace se = ::stream_executor;

template <typename T>
absl::Status select_k_exec(int device_ordinal,
                           se::DeviceMemoryAllocator* allocator,
                           se::Stream* stream, se::DeviceMemoryBase data_in,
                           se::DeviceMemoryBase data_out,
                           se::DeviceMemoryBase indices_out,
                           std::uint32_t batch, std::uint32_t n,
                           std::uint32_t k) {
  return absl::UnimplementedError(
      "select_k_exec is not implemented on this platform");
}

// Explicit instantiations for supported dtypes.
template absl::Status select_k_exec<float>(int, se::DeviceMemoryAllocator*,
                                           se::Stream*, se::DeviceMemoryBase,
                                           se::DeviceMemoryBase,
                                           se::DeviceMemoryBase, std::uint32_t,
                                           std::uint32_t, std::uint32_t);

template absl::Status select_k_exec<::xla::bfloat16>(
    int, se::DeviceMemoryAllocator*, se::Stream*, se::DeviceMemoryBase,
    se::DeviceMemoryBase, se::DeviceMemoryBase, std::uint32_t, std::uint32_t,
    std::uint32_t);

}  // namespace xla::gpu

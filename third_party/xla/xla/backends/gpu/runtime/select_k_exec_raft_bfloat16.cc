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
#include "xla/backends/gpu/runtime/raft_select_k_utils.h"
#include "xla/backends/gpu/runtime/select_k_exec_raft_impl.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

namespace xla::gpu {
namespace raft_internal {

template <>
SelectAlgo choose_select_k_algorithm<nv_bfloat16>(uint32_t rows, uint32_t cols,
                                                  uint32_t k) {
  return ToRaftSelectAlgo(choose_select_k_algorithm_bfloat16(rows, cols, k));
}

}  // namespace raft_internal

// Explicit specializations for nv_bfloat16
template absl::Status select_k_exec<nv_bfloat16>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, std::uint32_t, std::uint32_t,
    std::uint32_t, se::DeviceAddressBase);

// Explicit specializations for xla::bfloat16
template <>
absl::Status select_k_exec<::xla::bfloat16>(
    int device_ordinal, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, se::DeviceAddressBase data_in,
    se::DeviceAddressBase data_out, se::DeviceAddressBase indices_out,
    std::uint32_t batch, std::uint32_t n, std::uint32_t k,
    se::DeviceAddressBase scratch_buffer) {
  // Sanity check: Eigen::bfloat16 and nv_bfloat16 must be binary-compatible
  static_assert(sizeof(::xla::bfloat16) == sizeof(nv_bfloat16),
                "xla::bfloat16 and nv_bfloat16 must have the same size");

  // Just forward to the nv_bfloat16 instantiation
  return select_k_exec<nv_bfloat16>(device_ordinal, allocator, stream, data_in,
                                    data_out, indices_out, batch, n, k,
                                    scratch_buffer);
}

}  // namespace xla::gpu

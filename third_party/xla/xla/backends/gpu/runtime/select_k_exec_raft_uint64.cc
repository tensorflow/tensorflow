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
#include "xla/backends/gpu/runtime/select_k_exec_raft_impl.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
namespace raft_internal {

template <>
SelectAlgo choose_select_k_algorithm<uint64_t>(uint32_t rows, uint32_t cols,
                                               uint32_t k) {
  if (k > 129) {
    if (cols > 20215) {
      if (rows > 1013) {
        return SelectAlgo::kRadix11bitsExtraPass;
      } else {
        return SelectAlgo::kRadix11bits;
      }
    } else {
      if (k > 256) {
        return SelectAlgo::kRadix8bits;
      } else {
        return SelectAlgo::kWarpFiltered;
      }
    }
  } else {
    if (k > 1) {
      if (cols > 22089) {
        return SelectAlgo::kWarpDistributedShm;
      } else {
        if (rows > 341) {
          return SelectAlgo::kWarpDistributedShm;
        } else {
          return SelectAlgo::kWarpImmediate;
        }
      }
    } else {
      return SelectAlgo::kWarpImmediate;
    }
  }
}

}  // namespace raft_internal

// Explicit instantiation for xla::uint64_t
template absl::Status select_k_exec<std::uint64_t>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, std::uint32_t, std::uint32_t,
    std::uint32_t);

}  // namespace xla::gpu

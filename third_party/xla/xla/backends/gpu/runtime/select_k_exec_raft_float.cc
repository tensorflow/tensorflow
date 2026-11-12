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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/raft_select_k_utils.h"
#include "xla/backends/gpu/runtime/select_k_exec_raft_impl.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
namespace raft_internal {
namespace {

static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kAuto) ==
              static_cast<uint8_t>(SelectAlgo::kAuto));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kRadix8bits) ==
              static_cast<uint8_t>(SelectAlgo::kRadix8bits));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kRadix11bits) ==
              static_cast<uint8_t>(SelectAlgo::kRadix11bits));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kRadix11bitsExtraPass) ==
              static_cast<uint8_t>(SelectAlgo::kRadix11bitsExtraPass));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kWarpAuto) ==
              static_cast<uint8_t>(SelectAlgo::kWarpAuto));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kWarpImmediate) ==
              static_cast<uint8_t>(SelectAlgo::kWarpImmediate));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kWarpFiltered) ==
              static_cast<uint8_t>(SelectAlgo::kWarpFiltered));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kWarpDistributed) ==
              static_cast<uint8_t>(SelectAlgo::kWarpDistributed));
static_assert(static_cast<uint8_t>(RaftSelectKAlgo::kWarpDistributedShm) ==
              static_cast<uint8_t>(SelectAlgo::kWarpDistributedShm));

SelectAlgo ToRaftSelectAlgo(RaftSelectKAlgo algo) {
  switch (algo) {
    case RaftSelectKAlgo::kAuto:
      return SelectAlgo::kAuto;
    case RaftSelectKAlgo::kRadix8bits:
      return SelectAlgo::kRadix8bits;
    case RaftSelectKAlgo::kRadix11bits:
      return SelectAlgo::kRadix11bits;
    case RaftSelectKAlgo::kRadix11bitsExtraPass:
      return SelectAlgo::kRadix11bitsExtraPass;
    case RaftSelectKAlgo::kWarpAuto:
      return SelectAlgo::kWarpAuto;
    case RaftSelectKAlgo::kWarpImmediate:
      return SelectAlgo::kWarpImmediate;
    case RaftSelectKAlgo::kWarpFiltered:
      return SelectAlgo::kWarpFiltered;
    case RaftSelectKAlgo::kWarpDistributed:
      return SelectAlgo::kWarpDistributed;
    case RaftSelectKAlgo::kWarpDistributedShm:
      return SelectAlgo::kWarpDistributedShm;
  }
  LOG(FATAL) << "Invalid RaftSelectKAlgo value: " << static_cast<int>(algo);
}

}  // namespace

template <>
SelectAlgo choose_select_k_algorithm<float>(uint32_t rows, uint32_t cols,
                                            uint32_t k) {
  return ToRaftSelectAlgo(choose_select_k_algorithm_float(rows, cols, k));
}

}  // namespace raft_internal

template absl::Status select_k_exec<float>(int, se::DeviceAddressAllocator*,
                                           se::Stream*, se::DeviceAddressBase,
                                           se::DeviceAddressBase,
                                           se::DeviceAddressBase, std::uint32_t,
                                           std::uint32_t, std::uint32_t,
                                           se::DeviceAddressBase);

}  // namespace xla::gpu

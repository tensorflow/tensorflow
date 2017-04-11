/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/device_description.h"

#include <algorithm>

#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace perftools {
namespace gputools {

static const uint64 kUninitializedUint64 = -1ULL;
/* static */ const char *DeviceDescription::kUndefinedString = "<undefined>";

DeviceDescription::DeviceDescription()
    : device_vendor_(kUndefinedString),
      platform_version_(kUndefinedString),
      driver_version_(kUndefinedString),
      runtime_version_(kUndefinedString),
      pci_bus_id_(kUndefinedString),
      name_(kUndefinedString),
      thread_dim_limit_(kUninitializedUint64, kUninitializedUint64,
                        kUninitializedUint64),
      block_dim_limit_(kUninitializedUint64, kUninitializedUint64,
                       kUninitializedUint64),
      blocks_per_core_limit_(kUninitializedUint64),
      threads_per_core_limit_(kUninitializedUint64),
      threads_per_block_limit_(kUninitializedUint64),
      threads_per_warp_(kUninitializedUint64),
      registers_per_core_limit_(kUninitializedUint64),
      registers_per_block_limit_(kUninitializedUint64),
      registers_per_thread_limit_(kUninitializedUint64),
      warp_alloc_granularity_(1),
      register_alloc_granularity_(1),
      shared_memory_alloc_granularity_(1),
      device_address_bits_(kUninitializedUint64),
      device_memory_size_(kUninitializedUint64),
      shared_memory_per_core_(kUninitializedUint64),
      shared_memory_per_block_(kUninitializedUint64),
      clock_rate_ghz_(-1.0),
      cuda_compute_capability_major_(-1),
      cuda_compute_capability_minor_(-1),
      numa_node_(-1),
      core_count_(-1),
      ecc_enabled_(false) {}

std::unique_ptr<std::map<string, string>> DeviceDescription::ToMap() const {
  std::unique_ptr<std::map<string, string>> owned_result{
      new std::map<string, string>};
  std::map<string, string> &result = *owned_result;
  result["Device Vendor"] = device_vendor();
  result["Platform Version"] = platform_version();
  result["Driver Version"] = driver_version();
  result["Runtime Version"] = runtime_version();
  result["PCI bus ID"] = pci_bus_id_;
  result["Device Name"] = name_;

  const ThreadDim &thread_dim = thread_dim_limit();
  result["ThreadDim Limit"] =
      port::StrCat(thread_dim.x, ",", thread_dim.y, ",", thread_dim.z);
  const BlockDim &block_dim = block_dim_limit();
  result["BlockDim Limit"] =
      port::StrCat(block_dim.x, ",", block_dim.y, ",", block_dim.z);

  result["Threads Per Core Limit"] = port::StrCat(threads_per_core_limit());
  result["Threads Per Block Limit"] = port::StrCat(threads_per_block_limit());
  result["Registers Per Block Limit"] =
      port::StrCat(registers_per_block_limit());

  result["Device Address Bits"] = port::StrCat(device_address_bits());
  result["Device Memory Size"] =
      port::HumanReadableNumBytes::ToString(device_memory_size());

  result["Shared Memory Per Core"] =
      port::HumanReadableNumBytes::ToString(shared_memory_per_core_);
  result["Shared Memory Per Block"] =
      port::HumanReadableNumBytes::ToString(shared_memory_per_block_);

  result["Clock Rate GHz"] = port::StrCat(clock_rate_ghz());

  result["CUDA Compute Capability"] = port::StrCat(
      cuda_compute_capability_major_, ".", cuda_compute_capability_minor_);

  result["NUMA Node"] = port::StrCat(numa_node());
  result["Core Count"] = port::StrCat(core_count());
  result["ECC Enabled"] = port::StrCat(ecc_enabled());
  return owned_result;
}

namespace internal {

DeviceDescriptionBuilder::DeviceDescriptionBuilder()
    : device_description_(new DeviceDescription) {}

}  // namespace internal

bool DeviceDescription::cuda_compute_capability(int *major, int *minor) const {
  *major = cuda_compute_capability_major_;
  *minor = cuda_compute_capability_minor_;
  return cuda_compute_capability_major_ != 0;
}

bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim) {
  auto total_threads = thread_dim.x * thread_dim.y * thread_dim.z;
  auto threads_per_block_limit = device_description.threads_per_block_limit();
  if (total_threads > threads_per_block_limit) {
    VLOG(2) << "exceeded total-thread-per-block limit: " << total_threads
            << " vs limit " << threads_per_block_limit;
    return false;
  }

  const auto &limit = device_description.thread_dim_limit();
  bool ok = thread_dim.x <= limit.x && thread_dim.y <= limit.y &&
            thread_dim.z <= limit.z;
  if (!ok) {
    VLOG(2) << "thread dim " << thread_dim.ToString()
            << " exceeds limit contraints of " << limit.ToString();
  }
  return ok;
}

uint64 DivideCeil(uint64 x, uint64 y) {
  return port::MathUtil::CeilOfRatio(x, y);
}

void CalculateDimensionality(const DeviceDescription &device_description,
                             uint64 element_count, uint64 *threads_per_block,
                             uint64 *block_count) {
  *threads_per_block = device_description.threads_per_block_limit();
  *block_count = DivideCeil(element_count, *threads_per_block);
  if (*block_count == 1) {
    CHECK_LE(element_count, *threads_per_block);
    *threads_per_block = element_count;
  }
}

// Round value up to a multiple of n.
static uint64 RoundUp(uint64 value, uint64 n) {
  return port::MathUtil::CeilOfRatio(value, n) * n;
}

// Round value down to a multiple of n.
static uint64 RoundDown(uint64 value, uint64 n) {
  return port::MathUtil::FloorOfRatio(value, n) * n;
}

uint64 CalculateOccupancy(const DeviceDescription &device_description,
                          uint64 registers_per_thread,
                          uint64 shared_memory_per_block,
                          const ThreadDim &thread_dims) {
  // Don't try to compute occupancy if necessary values are not initialized.
  uint64 required_fields[] =  { device_description.registers_per_thread_limit(),
                                device_description.threads_per_warp(),
                                device_description.warp_alloc_granularity(),
                                device_description.register_alloc_granularity(),
                                device_description.registers_per_block_limit(),
                                device_description.shared_memory_per_core(),
                                device_description.blocks_per_core_limit() };
  for (auto value : required_fields) {
    if (value == kUninitializedUint64) {
      return 0;
    }
  }

  if (registers_per_thread > device_description.registers_per_thread_limit()) {
    return 0;
  }

  uint64 warps_per_block =
      port::MathUtil::CeilOfRatio(thread_dims.x * thread_dims.y * thread_dims.z,
                                  device_description.threads_per_warp());

  // Warp resources are allocated at a particular granularity.  This value is
  // the effective number of warps for resource allocation purposes.
  uint64 alloc_warps_per_block =
      RoundUp(warps_per_block, device_description.warp_alloc_granularity());

  uint64 alloc_regs_per_warp =
      RoundUp(device_description.threads_per_warp() * registers_per_thread,
              device_description.register_alloc_granularity());
  uint64 regs_per_block = alloc_warps_per_block * alloc_regs_per_warp;
  uint64 reg_limit =
      device_description.registers_per_block_limit() / regs_per_block;

  uint64 alloc_smem_per_block = RoundUp(
      shared_memory_per_block,
      device_description.shared_memory_alloc_granularity());
  uint64 smem_limit = alloc_smem_per_block > 0 ?
      device_description.shared_memory_per_core() / alloc_smem_per_block :
      device_description.blocks_per_core_limit();

  uint64 thread_limit = device_description.threads_per_core_limit()
      / (warps_per_block  * device_description.threads_per_warp());

  return std::min({ device_description.blocks_per_core_limit(),
          reg_limit, smem_limit, thread_limit });
}

uint64 CalculateRegisterLimitForTargetOccupancy(
    const DeviceDescription &device_description, uint64 shared_memory_per_block,
    const ThreadDim &thread_dims, uint64 target_blocks_per_core) {
  // Linear search from maximum number of registers down until the target
  // blocks per SM is found.
  // TODO(meheff): Compute this using a closed form solution.
  int reg_step = device_description.register_alloc_granularity() /
      device_description.threads_per_warp();
  for (int r = device_description.registers_per_thread_limit(); r > 0;
       r = RoundDown(r - 1, reg_step)) {
    uint64 occupancy = CalculateOccupancy(
        device_description, r, shared_memory_per_block, thread_dims);
    if (occupancy >= target_blocks_per_core) {
      return r;
    }
  }
  return 0;
}


}  // namespace gputools
}  // namespace perftools

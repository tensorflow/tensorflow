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

#include "absl/strings/str_cat.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/mathutil.h"

namespace stream_executor {

static const uint64_t kUninitializedUint64 = -1ULL;
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
      threads_per_core_limit_(kUninitializedUint64),
      threads_per_block_limit_(kUninitializedUint64),
      threads_per_warp_(kUninitializedUint64),
      registers_per_core_limit_(kUninitializedUint64),
      registers_per_block_limit_(kUninitializedUint64),
      device_address_bits_(kUninitializedUint64),
      device_memory_size_(kUninitializedUint64),
      memory_bandwidth_(kUninitializedUint64),
      shared_memory_per_core_(kUninitializedUint64),
      shared_memory_per_block_(kUninitializedUint64),
      clock_rate_ghz_(-1.0),
      numa_node_(-1),
      core_count_(-1),
      ecc_enabled_(false) {}

std::unique_ptr<std::map<std::string, std::string>> DeviceDescription::ToMap()
    const {
  std::unique_ptr<std::map<std::string, std::string>> owned_result{
      new std::map<std::string, std::string>};
  std::map<std::string, std::string> &result = *owned_result;
  result["Device Vendor"] = device_vendor();
  result["Platform Version"] = platform_version();
  result["Driver Version"] = driver_version();
  result["Runtime Version"] = runtime_version();
  result["PCI bus ID"] = pci_bus_id_;
  result["Device Name"] = name_;

  const ThreadDim &thread_dim = thread_dim_limit();
  result["ThreadDim Limit"] =
      absl::StrCat(thread_dim.x, ",", thread_dim.y, ",", thread_dim.z);
  const BlockDim &block_dim = block_dim_limit();
  result["BlockDim Limit"] =
      absl::StrCat(block_dim.x, ",", block_dim.y, ",", block_dim.z);

  result["Threads Per Core Limit"] = absl::StrCat(threads_per_core_limit());
  result["Threads Per Block Limit"] = absl::StrCat(threads_per_block_limit());
  result["Registers Per Block Limit"] =
      absl::StrCat(registers_per_block_limit());

  result["Device Address Bits"] = absl::StrCat(device_address_bits());
  result["Device Memory Size"] =
      port::HumanReadableNumBytes::ToString(device_memory_size());
  result["Memory Bandwidth"] = absl::StrCat(
      port::HumanReadableNumBytes::ToString(memory_bandwidth_), "/s");

  result["Shared Memory Per Core"] =
      port::HumanReadableNumBytes::ToString(shared_memory_per_core_);
  result["Shared Memory Per Block"] =
      port::HumanReadableNumBytes::ToString(shared_memory_per_block_);

  result["Clock Rate GHz"] = absl::StrCat(clock_rate_ghz());

  result["CUDA Compute Capability"] = cuda_compute_capability().ToString();

  result["AMDGPU GCN Arch Name"] = rocm_compute_capability().gcn_arch_name();

  result["NUMA Node"] = absl::StrCat(numa_node());
  result["Core Count"] = absl::StrCat(core_count());
  result["ECC Enabled"] = absl::StrCat(ecc_enabled());
  return owned_result;
}

namespace internal {

DeviceDescriptionBuilder::DeviceDescriptionBuilder()
    : device_description_(new DeviceDescription) {}

}  // namespace internal

CudaComputeCapability DeviceDescription::cuda_compute_capability() const {
  return cuda_compute_capability_;
}

RocmComputeCapability DeviceDescription::rocm_compute_capability() const {
  return rocm_compute_capability_;
}

bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim) {
  const int64_t total_threads = thread_dim.x * thread_dim.y * thread_dim.z;
  const int64_t threads_per_block_limit =
      device_description.threads_per_block_limit();
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
            << " exceeds limit constraints of " << limit.ToString();
  }
  return ok;
}

uint64_t DivideCeil(uint64 x, uint64 y) {
  return port::MathUtil::CeilOfRatio(x, y);
}

void CalculateDimensionality(const DeviceDescription &device_description,
                             int64_t element_count, int64_t *threads_per_block,
                             int64_t *block_count) {
  *threads_per_block = device_description.threads_per_block_limit();
  *block_count = port::MathUtil::CeilOfRatio(element_count, *threads_per_block);
  if (*block_count == 1) {
    CHECK_LE(element_count, *threads_per_block);
    *threads_per_block = element_count;
  }
}

}  // namespace stream_executor

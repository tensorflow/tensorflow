/* Copyright 2016 The OpenXLA Authors.

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

/// Implementation of HostExecutor class [of those methods not defined in the
// class declaration].
#include "xla/stream_executor/host/host_executor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/host/host_event.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/stream_executor/host/host_stream_factory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/profile_utils/cpu_utils.h"
#include "tsl/platform/mem.h"

namespace stream_executor {
namespace host {

absl::Status HostExecutor::Init() {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> HostExecutor::LoadKernel(
    const KernelLoaderSpec& /*spec*/) {
  return absl::UnimplementedError("No method of loading host kernel provided");
}

bool HostExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
  if (mem_info.free == INT64_MAX || mem_info.total == INT64_MAX) {
    *free = -1;
    *total = -1;
    return false;
  }
  *free = mem_info.free;
  *total = mem_info.total;
  return true;
}

DeviceAddressBase HostExecutor::Allocate(uint64_t size, int64_t memory_space) {
  CHECK_EQ(memory_space, 0);
  void* ptr = tsl::port::AlignedMalloc(
      size, static_cast<std::align_val_t>(kHostAlignment));
  if (size > 0 && ptr == nullptr) {
    return DeviceAddressBase();
  }
  return DeviceAddressBase(ptr, size);
}

void HostExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem != nullptr && mem->opaque() != nullptr) {
    tsl::port::AlignedFree(mem->opaque());
  }
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
HostExecutor::HostMemoryAllocate(uint64_t size) {
  void* ptr = tsl::port::AlignedMalloc(
      size, static_cast<std::align_val_t>(kHostAlignment));
  if (size > 0 && ptr == nullptr) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Failed to allocate %u bytes of aligned host memory", size));
  }
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [](void* location, uint64_t /*size*/) {
        tsl::port::AlignedFree(location);
      });
}

absl::Status HostExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                             const void* host_src,
                                             uint64_t size) {
  if (device_dst == nullptr || device_dst->opaque() == nullptr ||
      host_src == nullptr) {
    if (size == 0) {
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Null pointer passed to SynchronousMemcpy");
  }
  std::memcpy(device_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status HostExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  if (host_dst == nullptr || device_src.opaque() == nullptr) {
    if (size == 0) {
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Null pointer passed to SynchronousMemcpy");
  }
  std::memcpy(host_dst, device_src.opaque(), size);
  return absl::OkStatus();
}

void HostExecutor::DeallocateStream(Stream* /*stream*/) {}

absl::StatusOr<std::unique_ptr<Event>> HostExecutor::CreateEvent() {
  return std::make_unique<HostEvent>();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
HostExecutor::CreateDeviceDescription(int /*device_ordinal*/) {
  DeviceDescription desc;

  desc.set_device_address_bits(64);

  // TODO: b/511236711 - How to report a value that's based in reality but
  // that doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  desc.set_device_memory_size(int64_t{4} * 1024 * 1024 * 1024);

  int64_t cycle_counter_frequency =
      tsl::profile_utils::CpuUtils::GetCycleCounterFrequency();
  if (cycle_counter_frequency <= 0) {
    desc.set_clock_rate_ghz(
        1.0f);  // Fallback reasonable clock rate if unavailable
  } else {
    desc.set_clock_rate_ghz(static_cast<float>(cycle_counter_frequency) / 1e9f);
  }

  desc.set_name("Host");
  desc.set_platform_version("Default Version");

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<std::unique_ptr<Stream>> HostExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> /*priority*/) {
  std::shared_ptr<HostStreamFactory> factory = HostStreamFactory::GetFactory();
  if (factory != nullptr) {
    return factory->CreateStream(this);
  }
  return std::make_unique<HostStream>(this);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
HostExecutor::CreateMemoryAllocator(MemorySpace type) {
  if (type == MemorySpace::kHost) {
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size) { return HostMemoryAllocate(size); });
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", static_cast<int>(type)));
}

}  // namespace host
}  // namespace stream_executor

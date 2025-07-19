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

// Implementation of HostExecutor class [of those methods not defined in the
// class declaration].
#include "xla/stream_executor/host/host_executor.h"

#include <stdint.h>
#include <string.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
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
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/profile_utils/cpu_utils.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/mem.h"

namespace stream_executor {
namespace host {

HostStream* AsHostStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<HostStream*>(stream);
}

absl::Status HostExecutor::Init() {
  thread_pool_ = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "host-executor", tsl::port::MaxParallelism());
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> HostExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  return absl::InternalError("No method of loading host kernel provided");
}

bool HostExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

DeviceMemoryBase HostExecutor::Allocate(uint64_t size, int64_t memory_space) {
  CHECK_EQ(memory_space, 0);
  // Use a minimum alignment of 64 bytes to be friendly to AVX512 code.
  // This should probably be kept in sync with
  // tsl::Allocator::kAllocatorAlignment.
  return DeviceMemoryBase(
      tsl::port::AlignedMalloc(size, /*minimum_alignment=*/64), size);
}

void HostExecutor::Deallocate(DeviceMemoryBase* mem) {
  tsl::port::AlignedFree(mem->opaque());
}

absl::Status HostExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  memset(location->opaque(), 0, size);
  return absl::OkStatus();
}

absl::Status HostExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status HostExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return absl::OkStatus();
}

void HostExecutor::DeallocateStream(Stream* stream) {}

absl::StatusOr<std::unique_ptr<Event>> HostExecutor::CreateEvent() {
  return std::make_unique<HostEvent>();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
HostExecutor::CreateDeviceDescription(int device_ordinal) {
  DeviceDescription desc;

  desc.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  desc.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);

  float cycle_counter_frequency = static_cast<float>(
      tsl::profile_utils::CpuUtils::GetCycleCounterFrequency());
  desc.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

  desc.set_name("Host");
  desc.set_platform_version("Default Version");

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<std::unique_ptr<Stream>> HostExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  const HostStreamFactory* factory = HostStreamFactory::GetFactory();
  if (factory != nullptr) {
    return factory->CreateStream(this);
  }
  return std::make_unique<HostStream>(this);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
HostExecutor::CreateMemoryAllocator(MemoryType type) {
  if (type == MemoryType::kHost) {
    return std::make_unique<GenericMemoryAllocator>(
        [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          void* ptr = new char[size];
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [](void* location, uint64_t size) {
                delete[] static_cast<char*>(location);
              });
        });
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", type));
}

}  // namespace host
}  // namespace stream_executor

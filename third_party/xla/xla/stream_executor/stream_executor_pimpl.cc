/* Copyright 2015 The OpenXLA Authors.

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

// Implements the StreamExecutor interface by passing through to its
// implementation_ value (in pointer-to-implementation style), which
// implements StreamExecutorInterface.

#include "xla/stream_executor/stream_executor_pimpl.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_interface.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

// Get per-device memory limit in bytes. Returns 0 if
// TF_PER_DEVICE_MEMORY_LIMIT_MB environment variable is not set.
static int64_t GetMemoryLimitBytes() {
  int64_t value;
  TF_CHECK_OK(
      tsl::ReadInt64FromEnvVar("TF_PER_DEVICE_MEMORY_LIMIT_MB", 0, &value));
  return value * (1ll << 20);
}

StreamExecutor::StreamExecutor(
    const Platform* platform,
    std::unique_ptr<StreamExecutorInterface> implementation)
    : platform_(platform),
      implementation_(std::move(implementation)),
      memory_limit_bytes_(GetMemoryLimitBytes()),
      allocator_(this) {}

absl::Status StreamExecutor::Init() { return implementation_->Init(); }

absl::Status StreamExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                       Kernel* kernel) {
  return implementation_->GetKernel(spec, kernel);
}

void StreamExecutor::UnloadKernel(const Kernel* kernel) {
  implementation_->UnloadKernel(kernel);
}

absl::Status StreamExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                        ModuleHandle* module_handle) {
  return implementation_->LoadModule(spec, module_handle);
}

bool StreamExecutor::UnloadModule(ModuleHandle module_handle) {
  return implementation_->UnloadModule(module_handle);
}

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
StreamExecutor::CreateOrShareConstant(Stream* stream,
                                      absl::Span<const uint8_t> content) {
  return implementation_->CreateOrShareConstant(stream, content);
}

void StreamExecutor::Deallocate(DeviceMemoryBase* mem) {
  implementation_->Deallocate(mem);
}

bool StreamExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  return implementation_->CanEnablePeerAccessTo(other->implementation_.get());
}

absl::Status StreamExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return implementation_->EnablePeerAccessTo(other->implementation_.get());
}

const DeviceDescription& StreamExecutor::GetDeviceDescription() const {
  absl::MutexLock lock(&mu_);
  if (device_description_ != nullptr) {
    return *device_description_;
  }

  device_description_ = CreateDeviceDescription();
  return *device_description_;
}

dnn::DnnSupport* StreamExecutor::AsDnn() { return implementation_->AsDnn(); }

blas::BlasSupport* StreamExecutor::AsBlas() {
  return implementation_->AsBlas();
}

fft::FftSupport* StreamExecutor::AsFft() { return implementation_->AsFft(); }

absl::Status StreamExecutor::Launch(Stream* stream,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims,
                                    const Kernel& kernel,
                                    const KernelArgs& args) {
  return implementation_->Launch(stream, thread_dims, block_dims, kernel, args);
}

absl::Status StreamExecutor::Launch(Stream* stream,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims,
                                    const ClusterDim& cluster_dims,
                                    const Kernel& kernel,
                                    const KernelArgs& args) {
  return implementation_->Launch(stream, thread_dims, block_dims, cluster_dims,
                                 kernel, args);
}

absl::Status StreamExecutor::Submit(Stream* stream,
                                    const CommandBuffer& command_buffer) {
  return implementation_->Submit(stream, command_buffer);
}

absl::Status StreamExecutor::BlockHostUntilDone(Stream* stream) {
  return implementation_->BlockHostUntilDone(stream);
}

absl::Status StreamExecutor::GetStatus(Stream* stream) {
  return implementation_->GetStatus(stream);
}

DeviceMemoryBase StreamExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (memory_limit_bytes_ > 0 &&
      static_cast<int64_t>(size) > memory_limit_bytes_) {
    LOG(WARNING) << "Not enough memory to allocate " << size << " on device "
                 << device_ordinal()
                 << " within provided limit.  limit=" << memory_limit_bytes_
                 << "]";
    return DeviceMemoryBase();
  }
  return implementation_->Allocate(size, memory_space);
}

absl::StatusOr<DeviceMemoryBase> StreamExecutor::GetUntypedSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  // If failed to get the symbol, opaque/bytes are unchanged. Initialize them to
  // be nullptr/0 for consistency with DeviceMemory semantics.
  void* opaque = nullptr;
  size_t bytes = 0;
  if (GetSymbol(symbol_name, module_handle, &opaque, &bytes)) {
    return DeviceMemoryBase(opaque, bytes);
  }

  return absl::NotFoundError(
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}

bool StreamExecutor::GetSymbol(const std::string& symbol_name,
                               ModuleHandle module_handle, void** mem,
                               size_t* bytes) {
  return implementation_->GetSymbol(symbol_name, module_handle, mem, bytes);
}

void* StreamExecutor::UnifiedMemoryAllocate(uint64_t bytes) {
  return implementation_->UnifiedMemoryAllocate(bytes);
}

void StreamExecutor::UnifiedMemoryDeallocate(void* location) {
  return implementation_->UnifiedMemoryDeallocate(location);
}

absl::StatusOr<void*> StreamExecutor::CollectiveMemoryAllocate(uint64_t bytes) {
  return implementation_->CollectiveMemoryAllocate(bytes);
}

absl::Status StreamExecutor::CollectiveMemoryDeallocate(void* location) {
  return implementation_->CollectiveMemoryDeallocate(location);
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
StreamExecutor::HostMemoryAllocate(uint64_t size) {
  return implementation_->HostMemoryAllocate(size);
}

void StreamExecutor::HostMemoryDeallocate(void* data, uint64_t size) {
  return implementation_->HostMemoryDeallocate(data);
}

bool StreamExecutor::SynchronizeAllActivity() {
  return implementation_->SynchronizeAllActivity();
}

absl::Status StreamExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                                uint64_t size) {
  return implementation_->SynchronousMemZero(location, size);
}

absl::Status StreamExecutor::SynchronousMemcpyD2H(
    const DeviceMemoryBase& device_src, int64_t size, void* host_dst) {
  return implementation_->SynchronousMemcpy(host_dst, device_src, size);
}

absl::Status StreamExecutor::SynchronousMemcpyH2D(
    const void* host_src, int64_t size, DeviceMemoryBase* device_dst) {
  return implementation_->SynchronousMemcpy(device_dst, host_src, size);
}

bool StreamExecutor::Memcpy(Stream* stream, void* host_dst,
                            const DeviceMemoryBase& device_src, uint64_t size) {
  return implementation_->Memcpy(stream, host_dst, device_src, size).ok();
}

bool StreamExecutor::Memcpy(Stream* stream, DeviceMemoryBase* device_dst,
                            const void* host_src, uint64_t size) {
  return implementation_->Memcpy(stream, device_dst, host_src, size).ok();
}

bool StreamExecutor::MemcpyDeviceToDevice(Stream* stream,
                                          DeviceMemoryBase* device_dst,
                                          const DeviceMemoryBase& device_src,
                                          uint64_t size) {
  return implementation_->MemcpyDeviceToDevice(stream, device_dst, device_src,
                                               size);
}

absl::Status StreamExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                     uint64_t size) {
  return implementation_->MemZero(stream, location, size);
}

absl::Status StreamExecutor::Memset32(Stream* stream,
                                      DeviceMemoryBase* location,
                                      uint32_t pattern, uint64_t size) {
  return implementation_->Memset32(stream, location, pattern, size);
}

bool StreamExecutor::HostCallback(
    Stream* stream, absl::AnyInvocable<absl::Status() &&> callback) {
  return implementation_->HostCallback(stream, std::move(callback));
}

absl::Status StreamExecutor::AllocateEvent(Event* event) {
  return implementation_->AllocateEvent(event);
}

absl::Status StreamExecutor::DeallocateEvent(Event* event) {
  return implementation_->DeallocateEvent(event);
}

absl::Status StreamExecutor::RecordEvent(Stream* stream, Event* event) {
  return implementation_->RecordEvent(stream, event);
}

absl::Status StreamExecutor::WaitForEvent(Stream* stream, Event* event) {
  return implementation_->WaitForEvent(stream, event);
}

absl::Status StreamExecutor::WaitForEventOnExternalStream(std::intptr_t stream,
                                                          Event* event) {
  return implementation_->WaitForEventOnExternalStream(stream, event);
}

Event::Status StreamExecutor::PollForEventStatus(Event* event) {
  return implementation_->PollForEventStatus(event);
}

absl::StatusOr<std::unique_ptr<Stream>> StreamExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  auto stream = std::make_unique<Stream>(this);
  TF_RETURN_IF_ERROR(stream->Initialize(priority));
  return std::move(stream);
}

bool StreamExecutor::AllocateStream(Stream* stream) {
  return implementation_->AllocateStream(stream);
}

void StreamExecutor::DeallocateStream(Stream* stream) {
  implementation_->DeallocateStream(stream);
}

bool StreamExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  return implementation_->CreateStreamDependency(dependent, other);
}

std::unique_ptr<DeviceDescription> StreamExecutor::CreateDeviceDescription()
    const {
  return implementation_->CreateDeviceDescription().value();
}

bool StreamExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  return implementation_->DeviceMemoryUsage(free, total);
}

std::optional<AllocatorStats> StreamExecutor::GetAllocatorStats() {
  return implementation_->GetAllocatorStats();
}

bool StreamExecutor::ClearAllocatorStats() {
  return implementation_->ClearAllocatorStats();
}

Stream* StreamExecutor::FindAllocatedStream(void* gpu_stream) {
  return implementation_->FindAllocatedStream(gpu_stream);
}

StreamExecutorInterface* StreamExecutor::implementation() {
  return implementation_.get();
}

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    StreamExecutor* executor)
    : DeviceMemoryAllocator(executor->platform()) {
  stream_executors_ = {executor};
}

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    const Platform* platform,
    absl::Span<StreamExecutor* const> stream_executors)
    : DeviceMemoryAllocator(platform),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {}

absl::StatusOr<OwningDeviceMemory> StreamExecutorMemoryAllocator::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));
  DeviceMemoryBase result =
      executor->AllocateArray<uint8_t>(size, memory_space);
  if (size > 0 && result == nullptr) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Failed to allocate request for %s (%uB) on device ordinal %d",
        tsl::strings::HumanReadableNumBytes(size), size, device_ordinal));
  }
  VLOG(3) << absl::StreamFormat("Allocated %s (%uB) on device ordinal %d: %p",
                                tsl::strings::HumanReadableNumBytes(size), size,
                                device_ordinal, result.opaque());
  return OwningDeviceMemory(result, device_ordinal, this);
}

absl::Status StreamExecutorMemoryAllocator::Deallocate(int device_ordinal,
                                                       DeviceMemoryBase mem) {
  if (!mem.is_null()) {
    TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                        GetStreamExecutor(device_ordinal));
    VLOG(3) << absl::StreamFormat("Freeing %p on device ordinal %d",
                                  mem.opaque(), device_ordinal);
    executor->Deallocate(&mem);
  }
  return absl::OkStatus();
}

absl::StatusOr<StreamExecutor*>
StreamExecutorMemoryAllocator::GetStreamExecutor(int device_ordinal) const {
  if (device_ordinal < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "device ordinal value (%d) must be non-negative", device_ordinal));
  }
  for (StreamExecutor* se : stream_executors_) {
    if (se->device_ordinal() == device_ordinal) {
      return se;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Device %s:%d present but not supported",
                      platform()->Name(), device_ordinal));
}

bool StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation() const {
  return false;
}

absl::StatusOr<Stream*> StreamExecutorMemoryAllocator::GetStream(
    int device_ordinal) {
  CHECK(!AllowsAsynchronousDeallocation())
      << "The logic below only works for synchronous allocators";
  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));
  absl::MutexLock lock(&mutex_);
  if (!streams_.count(device_ordinal)) {
    auto p = streams_.emplace(std::piecewise_construct,
                              std::forward_as_tuple(device_ordinal),
                              std::forward_as_tuple(executor));
    TF_RETURN_IF_ERROR(p.first->second.Initialize());
    return &p.first->second;
  }
  return &streams_.at(device_ordinal);
}

}  // namespace stream_executor

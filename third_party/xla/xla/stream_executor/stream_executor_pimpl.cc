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
#include "xla/stream_executor/device_options.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/util/env_var.h"

namespace stream_executor {
namespace {

std::string StackTraceIfVLOG10() {
  if (VLOG_IS_ON(10)) {
    return absl::StrCat(" ", tsl::CurrentStackTrace(), "\n");
  } else {
    return "";
  }
}

}  // namespace

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
    std::unique_ptr<internal::StreamExecutorInterface> implementation,
    int device_ordinal)
    : platform_(platform),
      implementation_(std::move(implementation)),
      device_ordinal_(device_ordinal),
      live_stream_count_(0),
      memory_limit_bytes_(GetMemoryLimitBytes()),
      allocator_(this) {}

StreamExecutor::~StreamExecutor() {
  if (live_stream_count_.load() != 0) {
    LOG(WARNING) << "Not all streams were deallocated at executor destruction "
                 << "time. This may lead to unexpected/bad behavior - "
                 << "especially if any stream is still active!";
  }
}

StreamExecutor::PlatformSpecificHandle
StreamExecutor::platform_specific_handle() const {
  PlatformSpecificHandle handle;
  handle.context = implementation_->platform_specific_context();
  return handle;
}

absl::Status StreamExecutor::Init(DeviceOptions device_options) {
  TF_RETURN_IF_ERROR(
      implementation_->Init(device_ordinal_, std::move(device_options)));
  return absl::OkStatus();
}

absl::Status StreamExecutor::Init() { return Init(DeviceOptions::Default()); }

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
  VLOG(1) << "Called StreamExecutor::Deallocate(mem=" << mem->opaque()
          << ") mem->size()=" << mem->size() << StackTraceIfVLOG10();

  implementation_->Deallocate(mem);
  mem->Reset(nullptr, 0);
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

int64_t StreamExecutor::GetDeviceLoad() const {
  return implementation_->GetDeviceLoad();
}

dnn::DnnSupport* StreamExecutor::AsDnn() {
  absl::MutexLock lock(&mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }

  dnn_.reset(implementation_->CreateDnn());
  return dnn_.get();
}

blas::BlasSupport* StreamExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
  if (blas_ != nullptr) {
    return blas_.get();
  }

  blas_.reset(implementation_->CreateBlas());
  return blas_.get();
}

fft::FftSupport* StreamExecutor::AsFft() {
  absl::MutexLock lock(&mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }

  fft_.reset(implementation_->CreateFft());
  return fft_.get();
}

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
  absl::Status result;
  result = implementation_->BlockHostUntilDone(stream);
  return result;
}

absl::Status StreamExecutor::GetStatus(Stream* stream) {
  return implementation_->GetStatus(stream);
}

DeviceMemoryBase StreamExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (memory_limit_bytes_ > 0 &&
      static_cast<int64_t>(size) > memory_limit_bytes_) {
    LOG(WARNING) << "Not enough memory to allocate " << size << " on device "
                 << device_ordinal_
                 << " within provided limit.  limit=" << memory_limit_bytes_
                 << "]";
    return DeviceMemoryBase();
  }
  DeviceMemoryBase buf = implementation_->Allocate(size, memory_space);
  VLOG(1) << "Called StreamExecutor::Allocate(size=" << size
          << ", memory_space=" << memory_space << ") returns " << buf.opaque()
          << StackTraceIfVLOG10();

  return buf;
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
  void* buffer = implementation_->UnifiedMemoryAllocate(bytes);
  VLOG(1) << "Called StreamExecutor::UnifiedMemoryAllocate(size=" << bytes
          << ") returns " << buffer << StackTraceIfVLOG10();
  return buffer;
}

void StreamExecutor::UnifiedMemoryDeallocate(void* location) {
  VLOG(1) << "Called StreamExecutor::UnifiedMemoryDeallocate(location="
          << location << ")" << StackTraceIfVLOG10();

  return implementation_->UnifiedMemoryDeallocate(location);
}

absl::StatusOr<void*> StreamExecutor::CollectiveMemoryAllocate(uint64_t bytes) {
  TF_ASSIGN_OR_RETURN(void* buffer,
                      implementation_->CollectiveMemoryAllocate(bytes));
  VLOG(1) << "Called StreamExecutor::CollectiveMemoryAllocate(size=" << bytes
          << ") returns " << buffer << StackTraceIfVLOG10();
  return buffer;
}

absl::Status StreamExecutor::CollectiveMemoryDeallocate(void* location) {
  VLOG(1) << "Called StreamExecutor::CollectiveMemoryDeallocate(location="
          << location << ")" << StackTraceIfVLOG10();

  return implementation_->CollectiveMemoryDeallocate(location);
}

absl::StatusOr<std::unique_ptr<HostMemoryAllocation>>
StreamExecutor::HostMemoryAllocate(uint64_t size) {
  void* buffer = implementation_->HostMemoryAllocate(size);
  VLOG(1) << "Called StreamExecutor::HostMemoryAllocate(size=" << size
          << ") returns " << buffer << StackTraceIfVLOG10();
  if (buffer == nullptr && size > 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to allocate HostMemory of size %d", size));
  }
  return std::make_unique<HostMemoryAllocation>(buffer, size, implementation());
}

void StreamExecutor::HostMemoryDeallocate(void* data, uint64_t size) {
  VLOG(1) << "Called StreamExecutor::HostMemoryDeallocate(data=" << data << ")"
          << StackTraceIfVLOG10();

  return implementation_->HostMemoryDeallocate(data);
}

bool StreamExecutor::SynchronizeAllActivity() {
  VLOG(1) << "Called StreamExecutor::SynchronizeAllActivity()"
          << StackTraceIfVLOG10();
  bool ok = implementation_->SynchronizeAllActivity();

  return ok;
}

absl::Status StreamExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                                uint64_t size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemZero(location=" << location
          << ", size=" << size << ")" << StackTraceIfVLOG10();

  return implementation_->SynchronousMemZero(location, size);
}

bool StreamExecutor::SynchronousMemcpy(DeviceMemoryBase* device_dst,
                                       const DeviceMemoryBase& device_src,
                                       uint64_t size) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpy(device_dst="
          << device_dst->opaque() << ", device_src=" << device_src.opaque()
          << ", size=" << size << ") D2D" << StackTraceIfVLOG10();

  absl::Status status = implementation_->SynchronousMemcpyDeviceToDevice(
      device_dst, device_src, size);
  if (!status.ok()) {
    LOG(ERROR) << "synchronous memcpy: " << status;
  }
  return status.ok();
}

absl::Status StreamExecutor::SynchronousMemcpyD2H(
    const DeviceMemoryBase& device_src, int64_t size, void* host_dst) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpyD2H(device_src="
          << device_src.opaque() << ", size=" << size
          << ", host_dst=" << host_dst << ")" << StackTraceIfVLOG10();

  absl::Status result =
      implementation_->SynchronousMemcpy(host_dst, device_src, size);
  if (!result.ok()) {
    result = absl::InternalError(absl::StrFormat(
        "failed to synchronously memcpy device-to-host: device "
        "%p to host %p size %d: %s",
        device_src.opaque(), host_dst, size, result.ToString()));
  }

  return result;
}

absl::Status StreamExecutor::SynchronousMemcpyH2D(
    const void* host_src, int64_t size, DeviceMemoryBase* device_dst) {
  VLOG(1) << "Called StreamExecutor::SynchronousMemcpyH2D(host_src=" << host_src
          << ", size=" << size << ", device_dst=" << device_dst->opaque() << ")"
          << StackTraceIfVLOG10();

  absl::Status result =
      implementation_->SynchronousMemcpy(device_dst, host_src, size);
  if (!result.ok()) {
    result = absl::InternalError(absl::StrFormat(
        "failed to synchronously memcpy host-to-device: host "
        "%p to device %p size %d: %s",
        host_src, device_dst->opaque(), size, result.ToString()));
  }

  return result;
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
  CHECK_EQ(0, size % 4)
      << "need 32-bit multiple size to fill with 32-bit pattern";
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
  if (priority.has_value()) {
    if (std::holds_alternative<StreamPriority>(*priority)) {
      stream->SetPriority(std::get<StreamPriority>(*priority));
    } else {
      stream->SetPriority(std::get<int>(*priority));
    }
  }
  TF_RETURN_IF_ERROR(stream->Initialize());
  return std::move(stream);
}

bool StreamExecutor::AllocateStream(Stream* stream) {
  live_stream_count_.fetch_add(1, std::memory_order_relaxed);
  if (!implementation_->AllocateStream(stream)) {
    auto count = live_stream_count_.fetch_sub(1);
    CHECK_GE(count, 0) << "live stream count should not dip below zero";
    LOG(INFO) << "failed to allocate stream; live stream count: " << count;
    return false;
  }

  return true;
}

void StreamExecutor::DeallocateStream(Stream* stream) {
  dnn::DnnSupport* dnn;
  {
    absl::MutexLock lock(&mu_);
    dnn = dnn_.get();
  }
  if (dnn) {
    dnn->NotifyStreamDestroyed(stream);
  }
  implementation_->DeallocateStream(stream);
  CHECK_GE(live_stream_count_.fetch_sub(1), 0)
      << "live stream count should not dip below zero";
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

internal::StreamExecutorInterface* StreamExecutor::implementation() {
  return implementation_->GetUnderlyingExecutor();
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

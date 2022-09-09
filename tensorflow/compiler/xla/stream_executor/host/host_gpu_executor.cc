/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/stream_executor/host/host_gpu_executor.h"

#include <stdint.h>
#include <string.h>

#include <cstdint>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_stream.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_timer.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

namespace stream_executor {
namespace host {

HostStream* AsHostStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<HostStream*>(stream->implementation());
}

HostExecutor::HostExecutor(const PluginConfig& plugin_config)
    : plugin_config_(plugin_config) {}

HostExecutor::~HostExecutor() {}

port::Status HostExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  auto it =
      device_options.non_portable_tags.find("host_thread_stack_size_in_bytes");
  if (it != device_options.non_portable_tags.end()) {
    if (!absl::SimpleAtoi(it->second, &thread_stack_size_in_bytes_)) {
      return port::InvalidArgumentError(absl::StrCat(
          "Unable to parse host_thread_stack_size_in_bytes as an integer: ",
          it->second));
    }
  }
  return ::tensorflow::OkStatus();
}

bool HostExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  tensorflow::port::MemoryInfo mem_info = tensorflow::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

DeviceMemoryBase HostExecutor::Allocate(uint64_t size, int64_t memory_space) {
  CHECK_EQ(memory_space, 0);
  // Use a minimum alignment of 64 bytes to be friendly to AVX512 code.
  // This should probably be kept in sync with
  // tensorflow::Allocator::kAllocatorAlignment.
  return DeviceMemoryBase(
      tensorflow::port::AlignedMalloc(size, /*minimum_alignment=*/64), size);
}

void* HostExecutor::GetSubBuffer(DeviceMemoryBase* parent,
                                 uint64_t offset_bytes, uint64_t size_bytes) {
  return reinterpret_cast<char*>(parent->opaque()) + offset_bytes;
}

void HostExecutor::Deallocate(DeviceMemoryBase* mem) {
  tensorflow::port::AlignedFree(mem->opaque());
}

port::Status HostExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  memset(location->opaque(), 0, size);
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                             int value, uint64_t size) {
  memset(location->opaque(), value, size);
  return ::tensorflow::OkStatus();
}

bool HostExecutor::Memcpy(Stream* stream, void* host_dst,
                          const DeviceMemoryBase& gpu_src, uint64_t size) {
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  AsHostStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool HostExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                          const void* host_src, uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool HostExecutor::MemcpyDeviceToDevice(Stream* stream,
                                        DeviceMemoryBase* gpu_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

port::Status HostExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                   uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                  uint8 pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                    uint32_t pattern, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return ::tensorflow::OkStatus();
}

bool HostExecutor::HostCallback(Stream* stream,
                                std::function<port::Status()> callback) {
  AsHostStream(stream)->EnqueueTaskWithStatus(callback);
  return true;
}

bool HostExecutor::AllocateStream(Stream* stream) { return true; }

void HostExecutor::DeallocateStream(Stream* stream) {}

bool HostExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  auto event = std::make_shared<absl::Notification>();
  AsHostStream(other)->EnqueueTask([event]() { event->Notify(); });
  AsHostStream(dependent)->EnqueueTask(
      [event]() { event->WaitForNotification(); });
  return true;
}

class HostEvent : public internal::EventInterface {
 public:
  HostEvent() : notification_(std::make_shared<absl::Notification>()) {}

  std::shared_ptr<absl::Notification>& notification() { return notification_; }

 private:
  // We use a std::shared_ptr here because the client may delete the HostEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};

std::unique_ptr<internal::EventInterface>
HostExecutor::CreateEventImplementation() {
  return std::unique_ptr<internal::EventInterface>(new HostEvent());
}

static HostEvent* AsHostEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<HostEvent*>(event->implementation());
}

port::Status HostExecutor::AllocateEvent(Event* /*event*/) {
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::DeallocateEvent(Event* /*event*/) {
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::RecordEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask([notification]() {
    CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return ::tensorflow::OkStatus();
}

port::Status HostExecutor::WaitForEvent(Stream* stream, Event* event) {
  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask(
      [notification]() { notification->WaitForNotification(); });
  return ::tensorflow::OkStatus();
}

Event::Status HostExecutor::PollForEventStatus(Event* event) {
  absl::Notification& notification = *AsHostEvent(event)->notification();
  return notification.HasBeenNotified() ? Event::Status::kComplete
                                        : Event::Status::kPending;
}

bool HostExecutor::StartTimer(Stream* stream, Timer* timer) {
  dynamic_cast<HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool HostExecutor::StopTimer(Stream* stream, Timer* timer) {
  dynamic_cast<HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

port::Status HostExecutor::BlockHostUntilDone(Stream* stream) {
  return AsHostStream(stream)->BlockUntilDone();
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
HostExecutor::CreateDeviceDescription(int device_ordinal) {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);

  float cycle_counter_frequency = static_cast<float>(
      tensorflow::profile_utils::CpuUtils::GetCycleCounterFrequency());
  builder.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

  builder.set_name("Host");
  builder.set_platform_version("Default Version");

  return builder.Build();
}

bool HostExecutor::SupportsBlas() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                plugin_config_.blas())
      .ok();
}

blas::BlasSupport* HostExecutor::CreateBlas() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.value()(this);
}

bool HostExecutor::SupportsFft() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                               plugin_config_.fft())
      .ok();
}

fft::FftSupport* HostExecutor::CreateFft() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.value()(this);
}

bool HostExecutor::SupportsRng() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                               plugin_config_.rng())
      .ok();
}

rng::RngSupport* HostExecutor::CreateRng() {
  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.value()(this);
}

std::unique_ptr<internal::StreamInterface>
HostExecutor::GetStreamImplementation() {
  return std::unique_ptr<internal::StreamInterface>(
      new HostStream(thread_stack_size_in_bytes_));
}

}  // namespace host
}  // namespace stream_executor

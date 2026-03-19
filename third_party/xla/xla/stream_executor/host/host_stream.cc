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

// Class method definitions for HostStream, the Stream implementation for
// the HostExecutor implementation.
#include "xla/stream_executor/host/host_stream.h"

#include <string.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/host/host_event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
namespace host {

HostStream::HostStream(StreamExecutor* executor) : StreamCommon(executor) {}

HostStream::~HostStream() { parent()->DeallocateStream(this); }

absl::Status HostStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  memcpy(dst_mem, src_mem, size);
  return absl::OkStatus();
}

absl::Status HostStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  memcpy(host_dst, src_mem, size);
  return absl::OkStatus();
}

absl::Status HostStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  memcpy(dst_mem, host_src, size);
  return absl::OkStatus();
}

absl::Status HostStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  void* gpu_mem = location->opaque();
  memset(gpu_mem, pattern, size);
  return absl::OkStatus();
}

absl::Status HostStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  void* gpu_mem = location->opaque();
  memset(gpu_mem, 0, size);
  return absl::OkStatus();
}

absl::Status HostStream::WaitFor(Stream* other) { return absl::OkStatus(); }

absl::Status HostStream::WaitFor(Event* event) {
  std::shared_ptr<absl::Notification> notification =
      static_cast<HostEvent*>(event)->notification();
  notification->WaitForNotification();
  return absl::OkStatus();
}

absl::Status HostStream::RecordEvent(Event* event) {
  std::shared_ptr<absl::Notification> notification =
      static_cast<HostEvent*>(event)->notification();
  CHECK(!notification->HasBeenNotified());
  notification->Notify();
  return absl::OkStatus();
}

absl::Status HostStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  return std::move(callback)();
}

}  // namespace host
}  // namespace stream_executor

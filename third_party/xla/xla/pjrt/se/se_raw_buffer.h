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

#ifndef XLA_PJRT_SE_SE_RAW_BUFFER_H_
#define XLA_PJRT_SE_SE_RAW_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/pjrt/se/pjrt_stream_executor_client.h"
#include "xla/pjrt/se/tracked_device_buffer.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

class PjRtStreamExecutorDeviceEventPromise : public PjRtDeviceEventPromise {
 public:
  PjRtStreamExecutorDeviceEventPromise(PjRtStreamExecutorClient* client,
                                       LocalDeviceState* local_device,
                                       AsyncWorkRunner* async_work_runner);

  PjRtDeviceEventPtr event() const override {
    return PjRtDeviceEventPtr(event_.AsPtr());
  }

  void Set(PjRtDeviceEventRef event) override;

  void SetError(absl::Status s) override {
    av_->SetError(s);
    event_.SetError(std::move(s));
  }

  void SetFromSEEvent(BufferSequencingEventRef event);

  void SetReady() override;


  tsl::RCReference<tsl::IndirectAsyncValue>& av() { return av_; }

 private:
  PjRtStreamExecutorClient* client_;
  LocalDeviceState* local_device_;
  tsl::RCReference<tsl::IndirectAsyncValue> av_;
  tsl::AsyncValueRef<BufferSequencingEvent> event_;
};

class PjRtStreamExecutorRawBuffer : public CommonPjRtRawBufferImpl {
 public:
  PjRtStreamExecutorRawBuffer(
      PjRtStreamExecutorClient* client, PjRtMemorySpace* memory_space,
      LocalDeviceState* local_device,
      tsl::AsyncValueRef<RawSEDeviceMemory> device_buffer, size_t buffer_size)
      : client_(client),
        memory_space_(memory_space),
        local_device_(local_device),
        device_buffer_(device_buffer),
        buffer_size_(buffer_size) {}

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  LocalDeviceState* local_device() const { return local_device_; }

  absl::Status ValidateSlice(int64_t offset, int64_t slice_size);

  const tsl::AsyncValueRef<RawSEDeviceMemory>& device_buffer() const {
    return device_buffer_;
  }

  void* GetHostPointer() const override {
    return client_->IsOnCpu(memory_space()) ? device_buffer_->opaque()
                                            : nullptr;
  }

  void* OpaqueDeviceMemoryDataPointer() const override {
    return device_buffer_->opaque();
  }

  size_t GetOnDeviceSizeInBytes() const override { return buffer_size_; }

  absl::StatusOr<PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override;

  absl::StatusOr<PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override;

  absl::StatusOr<PjRtDeviceEventRef> MakeAllocationReadyEvent() override;

  absl::StatusOr<PjRtRawBufferRef> Slice(int64_t offset, int64_t size) override;

  void CopyTo(
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override;

  void ScheduleCopyTo(
      PjRtDeviceEventRefVector transfer_dependency_events,
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override;
  PjRtDeviceEventPtr GetRawBufferAsyncValue() override {
    return PjRtDeviceEventPtr::FromAsyncValue(device_buffer_.GetAsyncValue());
  }

  void DecrefAfter(PjRtDeviceEventRefVector avs) override { DropRef(); }

 private:
  PjRtStreamExecutorClient* client_;
  PjRtMemorySpace* memory_space_;
  LocalDeviceState* local_device_;
  tsl::AsyncValueRef<RawSEDeviceMemory> device_buffer_;
  size_t buffer_size_;

  void IntraClientCopyToWithDependencies(
      PjRtDeviceEventRefVector dependencies, PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event);
};

}  // namespace xla

#endif  // XLA_PJRT_SE_SE_RAW_BUFFER_H_

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

#ifndef XLA_PJRT_SE_RAW_BUFFER_H_
#define XLA_PJRT_SE_RAW_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/future.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

class PjRtStreamExecutorDeviceEvent : public PjRtDeviceEvent {
 public:
  explicit PjRtStreamExecutorDeviceEvent(
      tsl::AsyncValueRef<BufferSequencingEvent> event,
      const char* callee_type = "PjRtStreamExecutorDeviceEvent",
      const char* callee_method = "Unknown")
      : event_(std::move(event)),
        callee_type_(callee_type),
        callee_method_(callee_method) {}

  const tsl::AsyncValueRef<BufferSequencingEvent>& event() const {
    return event_;
  }

  tsl::AsyncValue* async_value() const override {
    return event_.GetAsyncValue();
  }

  Future<> GetReadyFuture() override;

 private:
  tsl::AsyncValueRef<BufferSequencingEvent> event_;
  const char* callee_type_;
  const char* callee_method_;
};

class PjRtStreamExecutorDeviceEventPromise : public PjRtDeviceEventPromise {
 public:
  PjRtStreamExecutorDeviceEventPromise(PjRtMemorySpace* memory_space,
                                       LocalDeviceState* local_device,
                                       AsyncWorkRunner* async_work_runner);

  tsl::AsyncValue* async_value() const override {
    return event_.GetAsyncValue();
  }

  void Set(tsl::RCReference<PjRtDeviceEvent> event) override;

  void SetError(absl::Status s) override {
    av_->SetError(s);
    event_.SetError(std::move(s));
  }

  void SetFromSEEvent(BufferSequencingEventRef event);

  void SetReady() override;

  const tsl::AsyncValueRef<BufferSequencingEvent>& event() const {
    return event_;
  }

  tsl::RCReference<tsl::IndirectAsyncValue>& av() { return av_; }

 private:
  PjRtMemorySpace* memory_space_;
  LocalDeviceState* local_device_;
  tsl::RCReference<tsl::IndirectAsyncValue> av_;
  tsl::AsyncValueRef<BufferSequencingEvent> event_;
};

class PjRtStreamExecutorDeviceEventSet : public PjRtDeviceEventSet {
 public:
  explicit PjRtStreamExecutorDeviceEventSet(size_t reservation) {
    events_.reserve(reservation);
  }

  void AddEvent(BufferSequencingEvent* event) { events_.insert(event); }

  const absl::flat_hash_set<BufferSequencingEvent*>& events() const {
    return events_;
  }

 private:
  absl::flat_hash_set<BufferSequencingEvent*> events_;
};

class PjRtStreamExecutorRawBuffer : public CommonPjRtRawBuffer {
 public:
  PjRtStreamExecutorRawBuffer(
      PjRtStreamExecutorClient* client, PjRtMemorySpace* memory_space,
      LocalDeviceState* local_device,
      tsl::AsyncValueRef<RawSEDeviceMemory> device_buffer)
      : client_(client),
        memory_space_(memory_space),
        local_device_(local_device),
        device_buffer_(device_buffer) {}

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  LocalDeviceState* local_device() const { return local_device_; }

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

  size_t GetOnDeviceSizeInBytes() const override {
    return device_buffer_->mem().size();
  }

  ShapedBuffer AsShapedBuffer(const xla::Shape&);

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  CopyRawHostToDeviceAndReturnEvent(const void* src, int64_t offset,
                                    int64_t transfer_size) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  CopyRawDeviceToHostAndReturnEvent(void* dst, int64_t offset,
                                    int64_t transfer_size) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> MakeAllocationReadyEvent()
      override;

  void ReadDynamicShape(tsl::AsyncValueRef<xla::Shape> output_shape,
                        xla::Shape shape) override;

  void CopyToLiteralAsync(
      Promise<> promise,
      tsl::RCReference<PjRtDeviceEventPromise> device_promise,
      MutableLiteralBase* literal, xla::Shape shape) override;
  void CopyTo(tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
              tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
              tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
              ::tsl::AsyncValueRef<bool> allocation_event) override;

 private:
  PjRtStreamExecutorClient* client_;
  PjRtMemorySpace* memory_space_;
  LocalDeviceState* local_device_;
  tsl::AsyncValueRef<RawSEDeviceMemory> device_buffer_;
};

}  // namespace xla

#endif  // XLA_PJRT_SE_RAW_BUFFER_H_

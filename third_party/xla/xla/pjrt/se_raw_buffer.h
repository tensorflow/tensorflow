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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/future.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/tsl/concurrency/async_value_ref.h"

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

class PjRtStreamExecutorDeviceEventSet : public PjRtDeviceEventSet {
 public:
  explicit PjRtStreamExecutorDeviceEventSet(size_t reservation) {
    events_.reserve(reservation);
  }

  void AddEvent(PjRtDeviceEventRef event) override;
  void AddEvent(const BufferSequencingEventRef& event);

  void AppendTo(
      std::vector<tsl::RCReference<tsl::AsyncValue>>& events) override;
  void AppendTo(std::vector<PjRtDeviceEventRef>& events) override;
  void AppendTo(PjRtDeviceEventSet& events) override;

  const absl::flat_hash_set<BufferSequencingEvent*>& events() const {
    return events_;
  }

  std::vector<BufferSequencingEventRef> event_refs() && {
    return std::move(event_refs_);
  }

 private:
  absl::flat_hash_set<BufferSequencingEvent*> events_;
  std::vector<BufferSequencingEventRef> event_refs_;
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

  ShapedBuffer AsShapedBuffer(const xla::Shape&);

  absl::StatusOr<PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size) override;

  absl::StatusOr<PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size) override;

  absl::StatusOr<PjRtDeviceEventRef> MakeAllocationReadyEvent() override;

  absl::StatusOr<PjRtRawBufferRef> Slice(int64_t offset, int64_t size) override;

  void ReadDynamicShape(tsl::AsyncValueRef<xla::Shape> output_shape,
                        xla::Shape shape) override;

  absl::StatusOr<PjRtRawBufferRef> RemoveDynamicShapeMetadataIfPresent(
      const xla::Shape& device_shape, const xla::Shape& logical_shape) override;

  void CopyToLiteralAsync(
      Promise<> promise,
      tsl::RCReference<PjRtDeviceEventPromise> device_promise,
      MutableLiteralBase* literal, xla::Shape shape) override;

  void CopyTo(PjRtRawBufferRef dst_raw_buffer,
              tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
              tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
              ::tsl::AsyncValueRef<bool> allocation_event) override;

  void ScheduleCopyTo(
      AsyncWorkRunner* async_work_runner,
      std::vector<PjRtDeviceEventRef> transfer_dependency_events,
      PjRtRawBufferRef dst_raw_buffer,
      tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
      tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
      ::tsl::AsyncValueRef<bool> allocation_event) override;
  tsl::AsyncValue* GetRawBufferAsyncValue() override {
    return device_buffer_.GetAsyncValue();
  }

  absl::StatusOr<PjRtDeviceEventRef> CopyRawToRemoteDevice(
      Future<std::string> serialized_descriptor, RemoteSendCallback on_done,
      std::vector<PjRtDeviceEventRef> transfer_dependency_avs) override;

  void DecrefAfter(std::vector<PjRtDeviceEventRef> avs) override { DropRef(); }

 private:
  PjRtStreamExecutorClient* client_;
  PjRtMemorySpace* memory_space_;
  LocalDeviceState* local_device_;
  tsl::AsyncValueRef<RawSEDeviceMemory> device_buffer_;
  size_t buffer_size_;

  void IntraClientCopyToWithDependencies(
      std::vector<PjRtDeviceEventRef> dependencies,
      PjRtRawBufferRef dst_raw_buffer,
      tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
      tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
      ::tsl::AsyncValueRef<bool> allocation_event);
};

}  // namespace xla

#endif  // XLA_PJRT_SE_RAW_BUFFER_H_

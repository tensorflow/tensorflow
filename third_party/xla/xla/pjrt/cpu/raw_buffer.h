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

#ifndef XLA_PJRT_CPU_RAW_BUFFER_H_
#define XLA_PJRT_CPU_RAW_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla {

class CpuTrackedDeviceEventPromise : public PjRtDeviceEventPromise {
 public:
  explicit CpuTrackedDeviceEventPromise(
      tsl::RCReference<tsl::IndirectAsyncValue> av)
      : av_(av) {}

  tsl::AsyncValue* async_value() const override { return av_.get(); }

  void Set(tsl::RCReference<PjRtDeviceEvent> event) override;

  void SetError(absl::Status s) override { av_->SetError(std::move(s)); }

  void SetReady() override;

  tsl::RCReference<tsl::IndirectAsyncValue>& av() { return av_; }

 private:
  tsl::RCReference<tsl::IndirectAsyncValue> av_;
};

class CpuTrackedDeviceEvent : public PjRtDeviceEvent {
 public:
  explicit CpuTrackedDeviceEvent(
      tsl::AsyncValueRef<CpuEvent> event,
      const char* callee_type = "CpuTrackedDeviceEvent",
      const char* callee_method = "Unknown")
      : event_(std::move(event)),
        callee_type_(callee_type),
        callee_method_(callee_method) {}

  const tsl::AsyncValueRef<CpuEvent>& event() const { return event_; }

  tsl::AsyncValue* async_value() const override {
    return event_.GetAsyncValue();
  }

  Future<> GetReadyFuture() override;

  static tsl::AsyncValueRef<CpuEvent> AfterAll(
      absl::Span<const tsl::RCReference<PjRtDeviceEvent>> events);

 private:
  tsl::AsyncValueRef<CpuEvent> event_;
  const char* callee_type_;
  const char* callee_method_;
};

class CpuTrackedDeviceEventSet : public PjRtDeviceEventSet {
 public:
  explicit CpuTrackedDeviceEventSet(size_t reservation) {
    events_.reserve(reservation);
  }

  void AddEvent(tsl::RCReference<tsl::AsyncValue> event) {
    events_.push_back(std::move(event));
  }

  absl::Span<const tsl::RCReference<tsl::AsyncValue>> events() const {
    return events_;
  }

  std::vector<tsl::RCReference<tsl::AsyncValue>> Consume() && {
    return std::move(events_);
  }

 private:
  std::vector<tsl::RCReference<tsl::AsyncValue>> events_;
};

class CpuRawBuffer : public CommonPjRtRawBuffer {
 public:
  CpuRawBuffer(PjRtMemorySpace* memory_space,
               tsl::AsyncValueRef<CpuDeviceMemory> buffer, size_t buffer_size,
               bool is_mutable)
      : memory_space_(memory_space),
        buffer_(std::move(buffer)),
        buffer_size_(buffer_size),
        is_mutable_(is_mutable) {}

  absl::Status ValidateSlice(int64_t offset, int64_t slice_size);

  // Allocates owning memory.
  static absl::StatusOr<tsl::RCReference<CpuRawBuffer>> Allocate(
      PjRtMemorySpace* memory_space, size_t size_bytes,
      const CpuDeviceMemory::Allocator& allocator =
          CpuDeviceMemory::DefaultAllocator());

  // Imports foreign memory.
  static absl::StatusOr<tsl::RCReference<CpuRawBuffer>> ImportForeignMemory(
      void* data, absl::AnyInvocable<void() &&> on_delete_callback,
      size_t on_device_bytes_count, PjRtMemorySpace* memory_space,
      bool is_mutable);

  size_t GetOnDeviceSizeInBytes() const override;

  void* GetHostPointer() const override;

  void* OpaqueDeviceMemoryDataPointer() const override {
    // We need to wait for the memory to be allocated before sharing it with
    // external frameworks like NumPy.
    tsl::BlockUntilReady(buffer_);
    CHECK(buffer_.IsConcrete());
    return buffer_->untyped_data();
  }

  const tsl::AsyncValueRef<CpuDeviceMemory>& buffer() const { return buffer_; }

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  bool is_mutable() const { return is_mutable_; }

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  CopyRawHostToDeviceAndReturnEvent(const void* src, int64_t offset,
                                    int64_t transfer_size) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  CopyRawDeviceToHostAndReturnEvent(void* dst, int64_t offset,
                                    int64_t transfer_size) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> CopyFromLiteral(
      const LiteralSlice& literal, const xla::Layout& layout,
      AsyncWorkRunner* async_work_runner);

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> MakeAllocationReadyEvent()
      override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> CopyFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      PjRtClient::HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const Shape& shape, AsyncWorkRunner* async_work_runner,
      absl::Mutex* transpose_mu, TransposePlanCache* transpose_cache,
      tsl::thread::ThreadPool* thread_pool, int max_transpose_threads);

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

  absl::StatusOr<tsl::RCReference<tsl::AsyncValue>> GetRawBufferAsyncValue()
      override {
    return buffer_.CopyRCRef();
  }

 private:
  PjRtMemorySpace* const memory_space_;
  tsl::AsyncValueRef<CpuDeviceMemory> buffer_;
  size_t buffer_size_;
  bool is_mutable_;
};

absl::StatusOr<xla::Shape> MakeDefaultCpuBufferShape(xla::Shape shape,
                                                     const xla::Layout* layout);

}  // namespace xla

#endif  // XLA_PJRT_CPU_RAW_BUFFER_H_

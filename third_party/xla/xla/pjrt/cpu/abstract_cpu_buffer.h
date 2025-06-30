/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_CPU_ABSTRACT_CPU_BUFFER_H_
#define XLA_PJRT_CPU_ABSTRACT_CPU_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/transpose.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A RAII helper class used to set an AsyncValueRef<CpuEvent> to a ready state
// upon destruction. In many cases in PjRt implementation, there will be
// multiple return statements in the function, all of which require setting some
// AsyncValueRef<CpuEvent> to be ready. This class could make such code more
// robust by using setting the AsyncValue in the destructor.
class MarkEventReadyOnExit {
 public:
  explicit MarkEventReadyOnExit(tsl::AsyncValueRef<CpuEvent> event)
      : event_(std::move(event)) {}

  MarkEventReadyOnExit(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit& operator=(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit(MarkEventReadyOnExit&&) noexcept = default;
  MarkEventReadyOnExit& operator=(MarkEventReadyOnExit&&) noexcept = default;

  ~MarkEventReadyOnExit() {
    if (event_) event_.SetStateConcrete();
  }

  tsl::AsyncValueRef<CpuEvent> Release() && { return std::move(event_); }

 private:
  tsl::AsyncValueRef<CpuEvent> event_;
};

class AbstractCpuBuffer : public CommonPjRtBuffer {
 public:
  class ScopedHold : public CommonPjRtBuffer::ScopedHold {
   public:
    TrackedCpuDeviceBuffer* buffer() const {
      return static_cast<TrackedCpuDeviceBuffer*>(
          CommonPjRtBuffer::ScopedHold::buffer());
    }
    TrackedCpuDeviceBuffer* operator->() const { return buffer(); }
    const TrackedCpuDeviceBuffer& operator*() const { return *buffer(); }
    AbstractCpuBuffer* parent() const {
      return static_cast<AbstractCpuBuffer*>(
          CommonPjRtBuffer::ScopedHold::parent());
    }

   private:
    using CommonPjRtBuffer::ScopedHold::ScopedHold;
    friend class AbstractCpuBuffer;
  };
  AbstractCpuBuffer(
      Shape on_device_shape,
      std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer,
      PjRtMemorySpace* memory_space);
  ~AbstractCpuBuffer() override;

  const Shape& on_device_shape() const override { return on_device_shape_; }

  absl::StatusOr<Shape> logical_on_device_shape() override;

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override {
    return PjRtFuture<>(Unimplemented("CopyRawToHost not implemented"));
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override {
    return Unimplemented("CopyToMemorySpace not implemented");
  }

  void Delete() override;

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override { return true; }

  // Acquires the device buffer for shared read-only usages, and it also adds
  // the `usage_event` to it. Any donation event in the future is expected to be
  // serialized after all the usage events added through this method. Returns
  // nullptr if the buffer is already donated or there is outstanding external
  // references.
  TrackedCpuDeviceBuffer* AcquireUsage(
      tsl::AsyncValueRef<CpuEvent> usage_event);

  // Acquires the device buffer for exclusive donation. The caller of this
  // method is expected to use the usage events and definition events to
  // serialize this donation with previous usages. After this method is called,
  // calls to AcquireUsage() will fail. Returns error status if the buffer is
  // already donated or there is outstanding external references.
  ScopedHold AcquireDonation();

  // Allocates a new `TrackedCpuDeviceBuffer` with the given shape and
  // definition events.
  static absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>>
  AllocateTrackedDeviceBuffer(
      const Shape& on_device_shape,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events);

  // Allocates new cpu events to `avs` and `definition_events`. If `shape` is a
  // tuple, multiple events will be allocated. Otherwise, `avs` and
  // `definition_events` will only contain one event.
  static void AllocateAvsAndEvents(
      const Shape& shape,
      absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>* definition_events);

  // A helper function to determine if a BufferFromHostBuffer call is elligable
  // for zero copy construction.
  static bool BufferFromHostBufferSupportsZeroCopy(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      const Shape& shape);

  // Returns a hold on the TrackedCpuDeviceBuffer holding the device
  // buffers. See comment on ScopedHold.
  ScopedHold GetBufferWithHold(ScopedHold::Type type);

 protected:
  virtual absl::string_view buffer_name() const = 0;

  TrackedCpuDeviceBuffer* device_buffer() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return static_cast<TrackedCpuDeviceBuffer*>(
        CommonPjRtBuffer::device_buffer());
  }

  PjRtFuture<> ToLiteralHelper(MutableLiteralBase* literal,
                               AsyncWorkRunner* async_work_runner);

  PjRtFuture<> DoAsyncWorkOnBuffer(
      absl::string_view method_name,
      absl::AnyInvocable<absl::Status(const Shape& device_shape,
                                      TrackedCpuDeviceBuffer* device_buffer) &&>
          work_on_buffer,
      bool should_do_work_sync, AsyncWorkRunner* async_work_runner);

  PjRtFuture<> CopyRawToHostHelper(void* dst, int64_t offset,
                                   int64_t transfer_size,
                                   AsyncWorkRunner* async_work_runner);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDeviceAcrossClients(
      PjRtDevice* dst_device);

  absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>> CopyToDeviceHelper(
      AsyncWorkRunner* async_work_runner);

  bool IsEmptyTuple() const {
    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes().size() == 0;
  }

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedCpuDeviceBuffer rather than freeing the device memory, so that
  // another framework can take ownership of it. The buffer returned from
  // Release may be safely dropped at any time even if it still has pending
  // async operations. The client should call Await before calling Release with
  // wait_for_operations_to_complete=false, to ensure that the host has
  // synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  const Shape on_device_shape_;
};

// Helper for copying into potentially sub-byte packed literals.
void PackOrCopy(PrimitiveType element_type, const LiteralSlice& literal,
                void* data, int64_t size);

}  // namespace xla

#endif  // XLA_PJRT_CPU_ABSTRACT_CPU_BUFFER_H_

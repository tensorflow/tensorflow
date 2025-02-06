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

#ifndef XLA_PJRT_CPU_ABSTRACT_TFRT_CPU_BUFFER_H_
#define XLA_PJRT_CPU_ABSTRACT_TFRT_CPU_BUFFER_H_

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
#include "xla/pjrt/cpu/tracked_tfrt_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/cpu/cpu_event.h"
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

// Async work runner abstracts away the implementation of the underlying thread
// pool (or concurrent work queue).
class AsyncWorkRunner {
 public:
  virtual ~AsyncWorkRunner() = default;

  // `work` euqueued by `Schedule` may run on the calling thread.
  virtual void Schedule(absl::AnyInvocable<void()> work) = 0;
  virtual void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void()> work) = 0;
};

class AbstractTfrtCpuBuffer : public PjRtBuffer {
 public:
  AbstractTfrtCpuBuffer(
      Shape on_device_shape,
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer);
  ~AbstractTfrtCpuBuffer() override;

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

  bool IsDeleted() override;

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  void CopyToRemoteDeviceScattered(
      PjRtFuture<std::vector<std::string>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const xla::PjRtBuffer::ScatterDetails& scatter_details) override {
    for (const auto& on_done : callbacks) {
      on_done(Unimplemented("Implement CopyToRemoteDeviceScattered."),
              /*sends_were_enqueued=*/false);
    }
  }

  PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override { return true; }

  // Acquires the device buffer for shared read-only usages, and it also adds
  // the `usage_event` to it. Any donation event in the future is expected to be
  // serialized after all the usage events added through this method. Returns
  // nullptr if the buffer is already donated or there is outstanding external
  // references.
  TrackedTfrtCpuDeviceBuffer* AcquireUsage(
      tsl::AsyncValueRef<CpuEvent> usage_event);

  // A helper class for managing a pending donation. It should be committed upon
  // success. Otherwise, the donated buffer is returned to the
  // AbstractTfrtCpuBuffer.
  class DonationTransaction {
   public:
    explicit DonationTransaction(
        AbstractTfrtCpuBuffer* buffer,
        std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer)
        : buffer_(buffer), device_buffer_(std::move(device_buffer)) {
      CHECK(buffer_);
    }
    DonationTransaction(const DonationTransaction&) = delete;
    DonationTransaction& operator=(const DonationTransaction&) = delete;
    DonationTransaction(DonationTransaction&&) = default;
    DonationTransaction& operator=(DonationTransaction&& other) noexcept {
      Abort();

      buffer_ = other.buffer_;
      device_buffer_ = std::move(other.device_buffer_);
      return *this;
    }

    ~DonationTransaction() { Abort(); }

    // Commit the donation. The rvalue ref qualifier is used to ensure the
    // semantic that it can be committed at most once.
    void Commit() && {
      buffer_->CommitDonation();
      device_buffer_.reset();
    }

    TrackedTfrtCpuDeviceBuffer* device_buffer() const {
      return device_buffer_.get();
    }

   private:
    void Abort() {
      if (device_buffer_) buffer_->AbortDonation(std::move(device_buffer_));
    }

    AbstractTfrtCpuBuffer* buffer_ = nullptr;
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer_;
  };

  // Acquires the device buffer for exclusive donation. The caller of this
  // method is expected to use the usage events and definition events to
  // serialize this donation with previous usages. After this method is called,
  // calls to AcquireUsage() will fail. Returns error status if the buffer is
  // already donated or there is outstanding external references.
  absl::StatusOr<DonationTransaction> AcquireDonation();

  // A helper function for PjRtClient::BufferFromHostLiteral. Copy the literal
  // to the current buffer asynchronously. `avs` is used to signal when the copy
  // is complete and `async_work_runner` is used to schedule the async work into
  // the underlying thread pool or work queue (usually owned by the client).
  void CopyFromLiteral(
      const LiteralSlice& literal, const Shape& shape,
      absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
      AsyncWorkRunner* async_work_runner);

  // Allocates a new `TrackedTfrtCpuDeviceBuffer` with the given shape and
  // definition events.
  static absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
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

  // A helper function for PjRtClient::BufferFromHostBuffer. Creates a new cpu
  // device buffer from the host buffer (maybe zero-copy or async).
  // `transpose_mu` and `transpose_cache` are used to transpose the input
  // layout.
  static absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
  BufferFromHostBufferHelper(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      PjRtClient::HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const Shape& shape, AsyncWorkRunner* async_work_runner,
      absl::Mutex* transpose_mu, TransposePlanCache* transpose_cache);

 protected:
  virtual absl::string_view buffer_name() const = 0;

  PjRtFuture<> ToLiteralHelper(MutableLiteralBase* literal,
                               AsyncWorkRunner* async_work_runner);

  PjRtFuture<> DoAsyncWorkOnBuffer(
      absl::string_view method_name,
      absl::AnyInvocable<
          absl::Status(const Shape& device_shape,
                       TrackedTfrtCpuDeviceBuffer* device_buffer) &&>
          work_on_buffer,
      bool should_do_work_sync, AsyncWorkRunner* async_work_runner);

  PjRtFuture<> CopyRawToHostHelper(void* dst, int64_t offset,
                                   int64_t transfer_size,
                                   AsyncWorkRunner* async_work_runner);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDeviceAcrossClients(
      PjRtDevice* dst_device);

  absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
  CopyToDeviceHelper(AsyncWorkRunner* async_work_runner);

  bool IsEmptyTuple() const {
    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes_size() == 0;
  }

  void DropExternalReference();

  // Commits the pending donation by setting `pending_donation_` to false.
  // `pending_donation_` must be true before calling this method.
  void CommitDonation();

  // Aborts the pending donation by returning the donated buffer, and setting
  // `pending_donation_` to false. `pending_donation_` must be true before
  // calling this method.
  void AbortDonation(std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer);

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedTfrtCpuDeviceBuffer rather than freeing the device memory, so that
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
  absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  // Releases the device buffer by returning a unique_ptr of it. If there is
  // outstanding donation or usage holds, this method blocks until those holds
  // are committed or dropped.
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> ReleaseBufferLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Shape on_device_shape_;

  mutable absl::Mutex mu_;
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of external references on the buffer.
  int external_reference_counter_ ABSL_GUARDED_BY(mu_) = 0;

  // If this buffer has external references when Delete() is called, this event
  // is populated by Delete(). When the last external reference is released,
  // the event is triggered, which is a precondition for the buffer being
  std::optional<tsl::AsyncValueRef<CpuEvent>> external_references_dropped_event_
      ABSL_GUARDED_BY(mu_);

  // `pending_donation_` indicates whether a donation is pending. The destructor
  // of the AbstractTfrtCpuBuffer will wait for a pending donation, as the
  // donation might fail. Note that concurrent calls to AcquireUsage() and
  // AcquireDonation() might fail even if the pending donation is aborted later.
  bool pending_donation_ ABSL_GUARDED_BY(mu_) = false;
};

class AbstractAsyncHostToHostMemoryTransferManager
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  ~AbstractAsyncHostToHostMemoryTransferManager() override;

  size_t buffer_count() const override { return buffer_sizes_.size(); }

  size_t buffer_size(int buffer_index) const override;

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override;

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override;

  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override;

  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override;

  void SetBufferError(int buffer_index, absl::Status error) override;

  void AddTransferMetadata(const TransferMetadata& meta) override {
    LOG(WARNING) << "AddTransferMetadata not implemented for "
                    "AbstractAsyncHostToHostMemoryTransferManager";
  }

 protected:
  AbstractAsyncHostToHostMemoryTransferManager(
      absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4> avs,
      absl::InlinedVector<std::unique_ptr<AbstractTfrtCpuBuffer>, 4> buffers,
      absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> device_buffers,
      absl::InlinedVector<size_t, 4> buffer_sizes,
      absl::InlinedVector<int64_t, 4> buffer_transfers_in_flight,
      absl::InlinedVector<bool, 4> last_transfer_finished,
      AsyncWorkRunner* async_work_runner);

  // Initialize `device_buffers`, `buffer_sizes`, `buffer_transfers_in_flight`,
  // and `last_transfer_finished` from `buffers`.
  static absl::Status PopulateAsyncTransferManagerData(
      absl::Span<const std::unique_ptr<AbstractTfrtCpuBuffer>> buffers,
      absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4>& device_buffers,
      absl::InlinedVector<size_t, 4>& buffer_sizes,
      absl::InlinedVector<int64_t, 4>& buffer_transfers_in_flight,
      absl::InlinedVector<bool, 4>& last_transfer_finished);

  absl::Status FillRawDataToSubBuffer(
      int buffer_index,
      absl::AnyInvocable<void(void* data, int64_t size)> fill_fn,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done);

  mutable absl::Mutex mu_;
  // The number of transfers that are currently in flight.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_);
  // AsyncValues used to mark buffers as ready for consumption.
  absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4> avs_
      ABSL_GUARDED_BY(mu_);
  // Holds the number of in-flight transfers for each buffer.
  absl::InlinedVector<int64_t, 4> buffer_transfers_in_flight_
      ABSL_GUARDED_BY(mu_);
  // Flag to indicate whether we have seen the last transfer of each buffer.
  absl::InlinedVector<bool, 4> last_transfer_finished_ ABSL_GUARDED_BY(mu_);
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<AbstractTfrtCpuBuffer>, 4> buffers_
      ABSL_GUARDED_BY(mu_);
  // Device buffers which we use to get the underlying memory to populate.
  absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> device_buffers_
      ABSL_GUARDED_BY(mu_);
  // Cached versions of the sizes of all the buffers. Not modified after
  // creation, so not guarded by mu_.
  absl::InlinedVector<size_t, 4> buffer_sizes_;

  AsyncWorkRunner* async_work_runner_;
};

}  // namespace xla

#endif  // XLA_PJRT_CPU_ABSTRACT_TFRT_CPU_BUFFER_H_

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

#ifndef XLA_PJRT_GPU_TFRT_TFRT_GPU_BUFFER_H_
#define XLA_PJRT_GPU_TFRT_TFRT_GPU_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
class TfrtGpuClient;
class TfrtGpuDevice;
class PjRtClient;
class PjRtDevice;

class TfrtGpuBuffer final : public PjRtBuffer {
 public:
  TfrtGpuBuffer(Shape on_device_shape,
                std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer,
                TfrtGpuClient* client, TfrtGpuDevice* device,
                PjRtMemorySpace* memory_space);
  ~TfrtGpuBuffer() override;

  TfrtGpuBuffer(const TfrtGpuBuffer&) = delete;
  TfrtGpuBuffer(TfrtGpuBuffer&&) = delete;
  TfrtGpuBuffer& operator=(const TfrtGpuBuffer&) = delete;
  TfrtGpuBuffer& operator=(TfrtGpuBuffer&&) = delete;

  PjRtMemorySpace* memory_space() const override { return memory_space_; }
  const Shape& on_device_shape() const override { return on_device_shape_; }
  PjRtDevice* device() const override;
  PjRtClient* client() const override;

  absl::StatusOr<Shape> logical_on_device_shape() override;

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  using PjRtBuffer::ToLiteralSync;
  Future<> ToLiteral(MutableLiteralBase* literal) override;

  Future<> LazyToLiteral(
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  Future<> CopyRawToHost(void* dst, int64_t offset,
                         int64_t transfer_size) override {
    return CopyRawToHostFuture(Future<void*>(dst), offset, transfer_size);
  }

  Future<> CopyRawToHostFuture(Future<void*> dst, int64_t offset,
                               int64_t transfer_size) override;

  void Delete() override;

  bool IsDeleted() const override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DonateWithControlDependency(
      Future<> dependency) override;

  Future<> GetReadyFuture() override;

  bool IsOnCpu() const override;

  const tsl::AsyncValueRef<GpuDeviceMemory>& GetBufferPtr() const;

 private:
  // Acquires the device buffer for shared read-only usages, and it also adds
  // the `usage_event` to it. Any donation event in the future is expected to be
  // serialized after all the usage events added through this method. Returns
  // nullptr if the buffer is already donated or there is outstanding external
  // references.
  TrackedGpuDeviceBuffer* AcquireUsage(
      tsl::AsyncValueRef<GpuEvent> usage_event);

  // A helper class for managing a pending donation. It should be committed upon
  // success. Otherwise, the donated buffer is returned to the TfrtGpuBuffer.
  class DonationTransaction {
   public:
    explicit DonationTransaction(
        tsl::AsyncValueRef<bool> donation_event,
        std::unique_ptr<TrackedGpuDeviceBuffer> device_buffer)
        : donation_event_(donation_event),
          device_buffer_(std::move(device_buffer)) {
      VLOG(3) << "DonationTransaction::DonationTransaction";
    }
    DonationTransaction(const DonationTransaction&) = delete;
    DonationTransaction& operator=(const DonationTransaction&) = delete;
    DonationTransaction(DonationTransaction&&) = default;
    DonationTransaction& operator=(DonationTransaction&& other) noexcept {
      Abort();

      donation_event_ = other.donation_event_;
      device_buffer_ = std::move(other.device_buffer_);
      return *this;
    }

    ~DonationTransaction() { Abort(); }

    // Commit the donation. The rvalue ref qualifier is used to ensure the
    // semantic that it can be committed at most once.
    void Commit() && {
      donation_event_.emplace(true);
      device_buffer_->SetUnOwned();
      device_buffer_.reset();
    }

    TrackedGpuDeviceBuffer* device_buffer() const {
      return device_buffer_.get();
    }

   private:
    void Abort() {
      if (device_buffer_) {
        VLOG(0) << "DonationTransaction::Abort is going to "
                   "abort donation: "
                << device_buffer_.get();
        donation_event_.emplace(false);
        device_buffer_.reset();  // TODO(b/382117736): We should put this back
                                 // into the TfrtGpuBuffer instead.
      }
    }

    tsl::AsyncValueRef<bool> donation_event_;
    std::unique_ptr<TrackedGpuDeviceBuffer> device_buffer_;
  };

  // Acquires the device buffer for exclusive donation. The caller of this
  // method is expected to use the usage events and definition events to
  // serialize this donation with previous usages. After this method is called,
  // calls to AcquireUsage() will fail. Returns error status if the buffer is
  // already donated or there is outstanding external references.
  absl::StatusOr<DonationTransaction> AcquireDonation()
      ABSL_LOCKS_EXCLUDED(mu_);

  tsl::AsyncValueRef<bool> GetDonationEvent() {
    absl::MutexLock lock(mu_);
    return donation_event_;
  }

  void DropExternalReference();

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedGpuDeviceBuffer rather than freeing the device memory, so that
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
  absl::StatusOr<std::unique_ptr<TrackedGpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  // Releases the device buffer by returning a unique_ptr of it.
  std::unique_ptr<TrackedGpuDeviceBuffer> ReleaseBufferLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Future<> ToLiteralHelper(
      MutableLiteralBase* literal,
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator);

  TfrtGpuClient* client_;
  const Shape on_device_shape_;
  TfrtGpuDevice* const device_;
  PjRtMemorySpace* const memory_space_;

  mutable absl::Mutex mu_;
  std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of external references on the buffer.
  int external_reference_counter_ ABSL_GUARDED_BY(mu_) = 0;

  // `pending_donation_` indicates whether a donation is pending. The destructor
  // of the TfrtGpuBuffer will wait for a pending donation, as the donation
  // might fail. Note that concurrent calls to AcquireUsage() and
  // AcquireDonation() might fail even if the pending donation is aborted later.
  tsl::AsyncValueRef<bool> donation_event_ ABSL_GUARDED_BY(mu_);
  Future<> ready_future_ ABSL_GUARDED_BY(mu_);

  // This event is triggered when the last external reference is released.
  // It is used to make sure that the buffer is not deleted before all external
  // references are dropped.
  // Notice that this event won't be triggered if there is never an external
  // reference.
  tsl::AsyncValueRef<GpuEvent> external_references_dropped_event_
      ABSL_GUARDED_BY(mu_);

  friend class TfrtGpuClient;
  friend class TfrtGpuExecutable;
  friend class DonationTransactionPeer;
};
}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_BUFFER_H_
